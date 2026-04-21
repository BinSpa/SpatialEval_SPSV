"""
inference_unified.py - Unified VLM inference for SpatialEval benchmark.

Supports modern VLMs: InternVL3, Qwen3-VL, Molmo2, LLaVA-OneVision-1.5, MiniCPM-V.

Usage:
    export HF_HUB_CACHE=/root/autodl-fs/models
    export HF_DATASETS_CACHE=/root/autodl-fs/datasets

    python inference_unified.py \
        --model_path "OpenGVLab/InternVL3_5-4B-HF" \
        --mode vqa --task all --w_reason \
        --output_folder outputs --device cuda

    # Or use local snapshot path directly:
    python inference_unified.py \
        --model_path "/root/autodl-fs/models/models--OpenGVLab--InternVL3_5-4B-HF/snapshots/XXX" \
        --model_name "OpenGVLab/InternVL3_5-4B-HF" \
        --mode vqa --task all --w_reason \
        --output_folder outputs --device cuda
"""

import argparse
import json
import os
import sys
import gc
from pathlib import Path
from copy import deepcopy
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from datasets import load_dataset


# ============================================================
# InternVL-style image preprocessing (for internvl_chat, sailvl)
# ============================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB')),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    """Dynamic preprocessing for InternVL-style models: split into tiles."""
    import math
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate target ratios
    target_ratios = set()
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j <= max_num and i * j >= min_num:
                    target_ratios.add((i, j))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find best aspect ratio
    best_ratio = (1, 1)
    best_diff = float('inf')
    for ratio in target_ratios:
        target = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target)
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio

    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]

    # Resize and split into tiles
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(best_ratio[0]):
        for j in range(best_ratio[1]):
            box = (i * image_size, j * image_size, (i + 1) * image_size, (j + 1) * image_size)
            processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        thumbnail = image.resize((image_size, image_size))
        processed_images.append(thumbnail)

    return processed_images


def preprocess_internvl_style(image, input_size=448, max_num=12):
    """Preprocess a PIL image into pixel_values tensor for InternVL/SAIL models."""
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Unified VLM inference for SpatialEval")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace model ID or local path to model snapshot.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Override model name for output filename (e.g. 'OpenGVLab/InternVL3_5-4B-HF').")
    parser.add_argument("--mode", choices=["tqa", "vqa", "vtqa"], required=True,
                        help="Input modality mode.")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "spatialmap", "mazenav", "spatialgrid", "spatialreal"],
                        help="Task to evaluate.")
    parser.add_argument("--dataset_id", type=str, default="MilaWang/SpatialEval",
                        help="HuggingFace dataset identifier.")
    parser.add_argument("--output_folder", type=str, default="outputs",
                        help="Output directory for results.")
    parser.add_argument("--w_reason", action="store_true",
                        help="Add chain-of-thought reasoning prompt.")
    parser.add_argument("--completion", action="store_true",
                        help="Add completion prompt.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--first_k", type=int, default=None,
                        help="Only process first k samples per question group.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference. Use 4-8 for significant speedup on GPUs.")
    parser.add_argument("--random_image", action="store_true",
                        help="Use random images instead of dataset images.")
    parser.add_argument("--noise_image", action="store_true",
                        help="Use noise images instead of dataset images.")
    return parser.parse_args()


# ============================================================
# Model type detection
# ============================================================

def resolve_model_path(model_path):
    """Resolve a HuggingFace model ID or local path to a local directory."""
    if os.path.isdir(model_path):
        return model_path
    # Try to resolve from HF cache
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(
            model_path,
            cache_dir=os.environ.get("HF_HUB_CACHE",
                                     os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")),
            local_files_only=True,
        )
    except Exception:
        pass
    # Fallback: try scanning the cache manually
    hub_cache = os.environ.get("HF_HUB_CACHE", "")
    if hub_cache:
        model_dir_name = "models--" + model_path.replace("/", "--")
        snapshot_dir = os.path.join(hub_cache, model_dir_name, "snapshots")
        if os.path.isdir(snapshot_dir):
            refs_dir = os.path.join(hub_cache, model_dir_name, "refs")
            ref = "main"
            if os.path.isdir(refs_dir):
                ref_files = os.listdir(refs_dir)
                if ref in ref_files:
                    with open(os.path.join(refs_dir, ref)) as f:
                        ref = f.read().strip()
            snapshot_path = os.path.join(snapshot_dir, ref)
            if os.path.isdir(snapshot_path):
                return snapshot_path
    raise FileNotFoundError(
        f"Could not resolve model path: {model_path}. "
        f"Provide either a local directory or a valid HuggingFace model ID with HF_HUB_CACHE set."
    )


def detect_model_type(model_path):
    """Detect model architecture type from config.json."""
    resolved = resolve_model_path(model_path)
    config_path = os.path.join(resolved, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found at {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    architectures = config.get("architectures", [])
    arch_str = str(architectures).lower()
    model_type = config.get("model_type", "").lower()

    # InternVL3-8B uses InternVLChatModel (custom code), while
    # InternVL3_5-4B uses InternVLForImageTextToText (native).
    if "internvlchatmodel" in arch_str or "internvl_forconditionalgeneration" in arch_str:
        # Old-style InternVL with model.chat() API
        return "internvl_chat"
    elif "internvl" in arch_str:
        return "internvl"
    elif "sailvl" in arch_str or model_type == "sailvl":
        return "sailvl"
    elif "qwen3vl" in arch_str:
        return "qwen3vl"
    elif "molmo" in arch_str:
        return "molmo2"
    elif "llava" in arch_str:
        return "llava_ov"
    elif "minicpmv" in arch_str:
        return "minicpmv"
    elif "gemma3" in arch_str or model_type == "gemma3":
        return "gemma3"
    else:
        raise ValueError(f"Unknown model architecture: {architectures} (model_type={model_type})")


# ============================================================
# Model loading
# ============================================================

def load_model_and_processor(model_type, model_path, device):
    """Load model and processor based on detected model type."""
    # Resolve to local path for config-based loading; keep original for from_pretrained
    resolved_path = resolve_model_path(model_path)

    if model_type == "internvl":
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(resolved_path)
        model = AutoModelForImageTextToText.from_pretrained(
            resolved_path, dtype=torch.bfloat16, device_map={"": 0}
        ).eval()

    elif model_type == "qwen3vl":
        from transformers import AutoProcessor, AutoModelForImageTextToText
        # Use smaller image resolution for faster inference
        min_pixels = 256 * 28 * 28
        max_pixels = 4 * 28 * 28 * 28
        processor = AutoProcessor.from_pretrained(
            resolved_path, min_pixels=min_pixels, max_pixels=max_pixels
        )
        model = AutoModelForImageTextToText.from_pretrained(
            resolved_path, dtype=torch.bfloat16, device_map={"": 0}
        ).eval()

    elif model_type == "molmo2":
        from transformers import AutoProcessor
        # Molmo2 maps to AutoModelForImageTextToText in its auto_map
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                resolved_path, dtype=torch.bfloat16, device_map={"": 0},
                trust_remote_code=True
            ).eval()
        except ImportError:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                resolved_path, dtype=torch.bfloat16, device_map={"": 0},
                trust_remote_code=True
            ).eval()
        processor = AutoProcessor.from_pretrained(resolved_path, trust_remote_code=True)

    elif model_type == "llava_ov":
        from transformers import AutoProcessor, AutoModelForCausalLM
        import transformers.modeling_flash_attention_utils as _fau
        # Compatibility: LLaVA-OV-1.5 custom code imports flash_attn_varlen_func
        # from transformers which was removed in transformers >= 4.50.
        # The actual function lives in the flash_attn package.
        if not hasattr(_fau, 'flash_attn_varlen_func'):
            try:
                from flash_attn import flash_attn_varlen_func as _favf
                _fau.flash_attn_varlen_func = _favf
            except ImportError:
                pass
        processor = AutoProcessor.from_pretrained(resolved_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            resolved_path, dtype=torch.bfloat16, device_map={"": 0},
            trust_remote_code=True
        ).eval()

    elif model_type == "minicpmv":
        from transformers import AutoProcessor, AutoModelForCausalLM
        processor = AutoProcessor.from_pretrained(resolved_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            resolved_path, dtype=torch.bfloat16, device_map={"": 0},
            trust_remote_code=True
        ).eval()

    elif model_type == "gemma3":
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(resolved_path)
        model = AutoModelForImageTextToText.from_pretrained(
            resolved_path, dtype=torch.bfloat16, device_map={"": 0}
        ).eval()

    elif model_type == "internvl_chat":
        # Old-style InternVL3 with custom code (model.chat() API)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_path, trust_remote_code=True, use_fast=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            resolved_path, dtype=torch.bfloat16, device_map={"": 0},
            trust_remote_code=True
        ).eval()
        return model, tokenizer

    elif model_type == "sailvl":
        # SAIL-VL2 (InternVL-style with Qwen3 backbone)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_path, trust_remote_code=True, use_fast=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            resolved_path, dtype=torch.bfloat16, device_map={"": 0},
            trust_remote_code=True
        ).eval()
        return model, tokenizer

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Set left-padding for correct batched generation with decoder-only models
    # Only set padding_side, do NOT set pad_token (it can trigger model re-init
    # and break custom models like LLaVA-OV-1.5 that lack pad_token_id in config)
    if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'padding_side'):
        processor.tokenizer.padding_side = "left"

    return model, processor


# ============================================================
# Inference per model type
# ============================================================

@torch.inference_mode()
def generate_standard(model, processor, question, image, args):
    """
    Standard generate for InternVL, Qwen3-VL, Molmo2, LLaVA-OV.
    Uses apply_chat_template + processor + model.generate.
    """
    # Build messages
    if image is not None:
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]}
        ]
    else:
        messages = [
            {"role": "user", "content": question}
        ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process inputs
    if image is not None:
        inputs = processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(model.device)
    else:
        inputs = processor(
            text=[text], images=None, return_tensors="pt"
        ).to(model.device)

    # Generate
    gen_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.temperature > 1e-5:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p
    else:
        gen_kwargs["do_sample"] = False

    output_ids = model.generate(**inputs, **gen_kwargs)

    # Decode only the generated part
    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    answer = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    return answer.strip()


@torch.inference_mode()
def generate_minicpmv(model, processor, question, image, args):
    """
    Generate for MiniCPM-V using its native chat() method.
    MiniCPM-V chat() expects a tokenizer (not processor) as the tokenizer arg.
    """
    msgs = [{"role": "user", "content": question}]

    # MiniCPM-V chat needs the actual tokenizer, not the processor
    tokenizer = None
    if hasattr(processor, 'tokenizer'):
        tokenizer = processor.tokenizer
    elif hasattr(processor, 'convert_tokens_to_ids'):
        tokenizer = processor
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            processor.name_or_path if hasattr(processor, 'name_or_path') else args.model_path,
            trust_remote_code=True
        )

    answer = model.chat(
        image=image,
        msgs=deepcopy(msgs),
        tokenizer=tokenizer,
        processor=processor,
        max_new_tokens=args.max_new_tokens,
        sampling=args.temperature > 1e-5,
        temperature=args.temperature if args.temperature > 1e-5 else 0.0,
    )

    return answer.strip()


@torch.inference_mode()
def generate_internvl_chat(model, tokenizer, question, image, args):
    """
    Generate for old-style InternVL3 / SAIL-VL2 using model.chat() API.
    These models use InternVL-style image preprocessing (dynamic tiling).
    The processor is actually a tokenizer for these models.
    """
    # Preprocess image into pixel_values
    if image is not None:
        pixel_values = preprocess_internvl_style(image).to(torch.bfloat16).to(model.device)
    else:
        pixel_values = None

    generation_config = {"max_new_tokens": args.max_new_tokens}
    if args.temperature > 1e-5:
        generation_config["do_sample"] = True
        generation_config["temperature"] = args.temperature
        generation_config["top_p"] = args.top_p
    else:
        generation_config["do_sample"] = False

    response = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=question,
        generation_config=generation_config,
    )

    return response.strip() if response else ""


@torch.inference_mode()
def generate_batch(model, processor, questions, images, args):
    """
    Batched generate for InternVL, Qwen3-VL, Molmo2, LLaVA-OV.
    Processes multiple samples in parallel for significant speedup.

    Returns list of answer strings, one per input.
    """
    batch_size = len(questions)
    texts = []

    for i in range(batch_size):
        if images[i] is not None:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": questions[i]}
            ]}]
        else:
            messages = [{"role": "user", "content": questions[i]}]

        texts.append(processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    # Process batch - prepare images per sample for the processor
    # Some processors (e.g., Gemma-3) expect a list-of-lists: [[img1], [img2], ...]
    # Others expect a flat list: [img1, img2, ...]
    # Detect by checking if processor raises on flat list vs nested list
    valid_images = [img for img in images if img is not None]
    if valid_images:
        try:
            inputs = processor(
                text=texts, images=valid_images, return_tensors="pt", padding=True
            ).to(model.device)
        except ValueError:
            # Likely Gemma-3 style: pass images as nested list
            nested_images = [[img] if img is not None else [] for img in images]
            inputs = processor(
                text=texts, images=nested_images, return_tensors="pt", padding=True
            ).to(model.device)
    else:
        inputs = processor(
            text=texts, return_tensors="pt", padding=True
        ).to(model.device)

    # Generate
    gen_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.temperature > 1e-5:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p
    else:
        gen_kwargs["do_sample"] = False

    output_ids = model.generate(**inputs, **gen_kwargs)

    # Decode each sequence
    input_len = inputs.input_ids.shape[1]
    answers = []
    for i in range(batch_size):
        generated = output_ids[i][input_len:]
        answers.append(processor.tokenizer.decode(generated, skip_special_tokens=True).strip())

    return answers


# ============================================================
# Prompt formatting
# ============================================================

def format_question(text, args):
    """Add reasoning or completion suffix to the question."""
    if args.w_reason:
        return (
            f"{text}\n"
            "First, provide a concise answer in one sentence. Then, "
            "elaborate on the reasoning behind your answer in a "
            "detailed, step-by-step explanation."
        )
    elif args.completion:
        return f"{text}\nAnswer:"
    else:
        return text


# ============================================================
# Image loading helpers
# ============================================================

def get_image(item, args):
    """Get image based on mode and image flags."""
    if args.mode == "tqa":
        return None
    if args.random_image:
        id_str = item["id"].lower()
        if "mazenav" in id_str:
            return Image.open("assets/random_maze_nav.png").convert("RGB")
        elif "spatialgrid" in id_str:
            return Image.open("assets/random_spatial_grid.png").convert("RGB")
        elif "spatialmap" in id_str:
            return Image.open("assets/random_spatial_map.png").convert("RGB")
        else:
            return Image.open("assets/noise.png").convert("RGB")
    elif args.noise_image:
        return Image.open("assets/noise.png").convert("RGB")
    else:
        return item["image"].convert("RGB") if item.get("image") else None


# ============================================================
# Output path formatting (compatible with evaluation.py)
# ============================================================

def format_output_path(args):
    """Build output filepath compatible with the existing evaluation script."""
    # Use model_name if provided, otherwise extract from model_path
    if args.model_name:
        model_name = args.model_name.replace("/", "__")
    else:
        model_name = args.model_path.replace("/", "__")

    output_dir = Path(args.output_folder) / args.dataset_id.replace("/", "__") / args.mode / args.task
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"m-{model_name}"
    if args.random_image:
        filename += "_random_image"
    elif args.noise_image:
        filename += "_noise_image"
    if args.w_reason:
        filename += "_w_reason"
    elif args.completion:
        filename += "_completion"
    else:
        filename += "_bare"

    return output_dir / f"{filename}.jsonl"


# ============================================================
# Main inference loop
# ============================================================

def run_inference(args, model, processor, model_type, dataset, output_path):
    """Run inference over the full dataset and write JSONL results."""

    use_batch = args.batch_size > 1 and model_type not in ("minicpmv", "internvl_chat", "sailvl")
    batch_size = args.batch_size if use_batch else 1

    # Flatten all items to process into a single list
    question_groups = {}
    for item in dataset:
        task = item["id"].split(".")[0]
        question_id = item["id"].split(".")[-1]
        group_key = f"{task}.{question_id}"
        if group_key not in question_groups:
            question_groups[group_key] = []
        question_groups[group_key].append(item)

    all_items = []
    for group_key, items in question_groups.items():
        num_to_process = args.first_k if args.first_k is not None else len(items)
        all_items.extend(items[:num_to_process])

    total = len(all_items)
    processed = 0

    with open(output_path, "w", encoding="utf-8") as outfile:
        for batch_start in range(0, total, batch_size):
            batch_items = all_items[batch_start:batch_start + batch_size]
            current_bs = len(batch_items)

            # Prepare batch inputs
            batch_questions = []
            batch_images = []
            batch_ids = []
            for item in batch_items:
                batch_ids.append(item["id"])
                batch_images.append(get_image(item, args))
                batch_questions.append(format_question(item["text"], args))

            # Generate
            try:
                if use_batch:
                    answers = generate_batch(
                        model, processor, batch_questions, batch_images, args
                    )
                elif model_type == "minicpmv":
                    answers = []
                    for i in range(current_bs):
                        ans = generate_minicpmv(
                            model, processor, batch_questions[i], batch_images[i], args
                        )
                        answers.append(ans)
                elif model_type in ("internvl_chat", "sailvl"):
                    answers = []
                    for i in range(current_bs):
                        ans = generate_internvl_chat(
                            model, processor, batch_questions[i], batch_images[i], args
                        )
                        answers.append(ans)
                else:
                    answers = []
                    for i in range(current_bs):
                        ans = generate_standard(
                            model, processor, batch_questions[i], batch_images[i], args
                        )
                        answers.append(ans)
            except Exception as e:
                print(f"[ERROR] Batch failed: {e}", file=sys.stderr)
                answers = [""] * current_bs

            # Write results
            for i in range(current_bs):
                image_path_str = ""
                if args.mode == "tqa":
                    image_path_str = ""
                elif args.random_image:
                    image_path_str = "random"
                elif args.noise_image:
                    image_path_str = "noise"
                else:
                    image_path_str = "dataset"

                result = {
                    "id": batch_ids[i],
                    "answer": answers[i] if i < len(answers) else "",
                    "oracle_answer": batch_items[i]["oracle_answer"],
                    "oracle_option": batch_items[i]["oracle_option"],
                    "oracle_full_answer": batch_items[i]["oracle_full_answer"],
                    "prompt": batch_questions[i],
                    "image": image_path_str,
                }
                outfile.write(json.dumps(result) + "\n")

            outfile.flush()
            os.fsync(outfile.fileno())

            processed += current_bs
            if processed % max(10, batch_size * 5) == 0 or processed == total:
                elapsed_info = f" (batch={current_bs})" if use_batch else ""
                print(f"[{processed}/{total}]{elapsed_info} Q: {batch_ids[-1]}")
                print(f"  A: {answers[-1][:120] if answers else ''}")

    print(f"Results saved to {output_path} ({processed} samples)")
    return output_path


# ============================================================
# Entry point
# ============================================================

def main():
    args = parse_args()

    print(f"Model: {args.model_path}")
    print(f"Mode: {args.mode}, Task: {args.task}, Reasoning: {args.w_reason}")

    # Detect model type
    model_type = detect_model_type(args.model_path)
    print(f"Detected model type: {model_type}")

    # Load dataset
    cache_dir = os.environ.get("HF_DATASETS_CACHE")
    dataset = load_dataset(args.dataset_id, args.mode, split="test", cache_dir=cache_dir)
    if args.task != "all":
        dataset = dataset.filter(lambda x: args.task in x["id"])
    print(f"Dataset: {len(dataset)} samples")

    # Load model
    print("Loading model...")
    model, processor = load_model_and_processor(model_type, args.model_path, args.device)
    print(f"Model loaded on {args.device}")

    # Quick smoke test
    print("Running smoke test...")
    test_item = dataset[0]
    test_image = get_image(test_item, args)
    test_question = format_question(test_item["text"], args)

    try:
        if model_type == "minicpmv":
            gen_fn = generate_minicpmv
        elif model_type in ("internvl_chat", "sailvl"):
            gen_fn = generate_internvl_chat
        else:
            gen_fn = generate_standard
        test_answer = gen_fn(model, processor, test_question, test_image, args)
        print(f"Smoke test passed. Answer: {test_answer[:100]}")
    except Exception as e:
        print(f"Smoke test FAILED: {e}")
        print("Attempting model-specific fallback...")
        # Will be handled per-model during the full run
        raise

    # Format output path
    output_path = format_output_path(args)
    print(f"Output: {output_path}")

    # Run full inference
    run_inference(args, model, processor, model_type, dataset, output_path)

    # Cleanup
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
