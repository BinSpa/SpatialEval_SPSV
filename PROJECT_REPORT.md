# SpatialEval VLM Evaluation - Project Report

## 1. Project Overview

This project evaluates **11 modern Vision-Language Models (VLMs)** on the **SpatialEval benchmark** (NeurIPS 2024), a spatial reasoning benchmark measuring how well VLMs understand spatial relationships in images.

### Benchmark Structure

| Mode | Description | Samples |
|------|-------------|---------|
| TQA | Text-Only Question Answering (no image) | 4,635 |
| VQA | Visual Question Answering (image provided) | 4,635 |
| VTQA | Visual + Text Question Answering | 4,635 |
| **Total** | | **13,905** |

### Tasks

| Task | Description | Samples | Question Types |
|------|-------------|---------|----------------|
| spatialmap | Direction/relative position on synthetic maps | 1,500 | qid0: direction, qid1: nearest object, qid2: count |
| mazenav | Path counting/navigation in mazes | 1,500 | qid0: count paths, qid1: count steps, qid2: yes/no reachability |
| spatialgrid | Animal identification/counting in 5x5 grids | 1,500 | qid0: count, qid1: identify, qid2: identify at position |
| spatialreal | Real-world image spatial reasoning | 135 | qid0: varied counting/identification |

### Models Evaluated

| Model | Architecture | Parameters | Loading Class |
|-------|-------------|-----------|---------------|
| InternVL3_5-4B-HF | InternVL | 4B | AutoModelForImageTextToText |
| InternVL3-8B-Instruct | InternVL (custom) | 8B | AutoModelForCausalLM (trust_remote_code, model.chat()) |
| Qwen3-VL-4B-Instruct | Qwen3VL | 4B | AutoModelForImageTextToText |
| Qwen3-VL-8B-Instruct | Qwen3VL | 8B | AutoModelForImageTextToText |
| Molmo2-4B | Molmo | 4B | AutoModelForImageTextToText |
| Molmo2-8B | Molmo | 8B | AutoModelForImageTextToText |
| LLaVA-OneVision-1.5-4B | LLaVA-OV | 4B | AutoModelForCausalLM (trust_remote_code) |
| LLaVA-OneVision-1.5-8B | LLaVA-OV | 8B | AutoModelForCausalLM (trust_remote_code) |
| MiniCPM-V-4.5 | MiniCPM-V | 4.5B | AutoModelForCausalLM (trust_remote_code, model.chat()) |
| SAIL-VL2-8B | SAIL-VL (custom) | 8B | AutoModelForCausalLM (trust_remote_code, model.chat()) |
| Gemma-3-4b-it | Gemma3 | 4B | AutoModelForImageTextToText |

### Hardware

- 2x NVIDIA A800 80GB GPUs
- PyTorch 2.8.0, Transformers 4.57.1, flash_attn 2.8.3
- Models cached at `/root/autodl-fs/models` (HF_HUB_CACHE)
- Datasets cached at `/root/autodl-fs/datasets` (HF_DATASETS_CACHE)

---

## 2. Project File Structure

```
SpatialEval_SPSV/
├── inference_unified.py          # [NEW] Unified VLM inference script (main contribution)
├── inference_vlm.py              # [FIXED] Original VLM inference (bug fix on line 139)
├── inference_lm.py               # Language model inference (unchanged)
├── evals/
│   └── evaluation.py             # [FIXED] Added spatialreal handler
├── scripts/
│   ├── run_full_eval.sh          # [NEW] Dual-GPU parallel evaluation runner
│   ├── run_all_evals.sh          # [NEW] Sequential evaluation runner
│   └── compare_results.py        # [NEW] Results comparison tool
├── outputs/
│   └── MilaWang__SpatialEval/vqa/all/
│       ├── m-Qwen__Qwen3-VL-4B-Instruct_w_reason.jsonl       (4635 lines)
│       ├── m-OpenGVLab__InternVL3_5-4B-HF_w_reason.jsonl     (4635 lines)
│       ├── m-allenai__Molmo2-4B_w_reason.jsonl                (4635 lines)
│       ├── m-lmms-lab__LLaVA-OneVision-1.5-4B-Instruct_w_reason.jsonl (4635 lines)
│       ├── m-lmms-lab__LLaVA-OneVision-1.5-8B-Instruct_w_reason.jsonl (4635 lines)
│       └── m-openbmb__MiniCPM-V-4_5_w_reason.jsonl            (4635 lines)
├── results_comparison.md         # [UPDATED] Full comparison results
├── PROJECT_REPORT.md             # [NEW] This file
├── assets/                       # Random/noise images for ablation
├── configs/                      # Model configurations
├── models/                       # Model-specific inference scripts
└── utils/                        # Utility functions
```

---

## 3. What Was Done

### 3.1 Created Unified Inference Script (`inference_unified.py`)

The original codebase had separate scripts for each model type with inconsistent APIs. We created a single unified script that:

- **Auto-detects model architecture** from `config.json` (InternVL, Qwen3-VL, Molmo2, LLaVA-OV, MiniCPM-V)
- **Resolves HuggingFace cache paths** automatically - handles both model IDs (`OpenGVLab/InternVL3_5-4B-HF`) and local snapshot paths
- **Supports batched inference** (batch_size=8 gives ~2.7x speedup for standard models)
- **Handles model-specific APIs**: MiniCPM-V uses `model.chat()` instead of `model.generate()`
- **Dual-GPU parallel execution**: Uses `CUDA_VISIBLE_DEVICES` + `device_map={"": 0}` to pin each model to a specific GPU
- **Robust error handling**: Catches per-batch failures without losing progress

### 3.2 Fixed Bugs in Existing Code

1. `inference_vlm.py` line 139: Python truthiness bug
2. `evals/evaluation.py`: Missing spatialreal task handler

### 3.3 Ran Full Evaluation

- All 6 models x 4635 samples = 27,810 total inference calls
- VQA mode with chain-of-thought reasoning (`--w_reason`)
- Dual-GPU parallel execution completed in ~8 hours

---

## 4. Problems Encountered and Solutions

### Problem 1: Python Truthiness Bug in Original Code

**File**: `inference_vlm.py`, line 139

**Symptom**: The condition `"qwen" or "cog" or "instructblip" or "llava"` always evaluates to `True` (Python truthiness - non-empty string `"qwen"` is truthy, `or` short-circuits).

**Fix**:
```python
# Before (always True):
if "qwen" or "cog" or "instructblip" or "llava" in args.model_path.lower():

# After (correct):
if any(k in args.model_path.lower() for k in ("qwen", "cog", "instructblip", "llava")):
```

**Impact**: This bug meant the original code always took the same inference branch regardless of model type, potentially producing incorrect results for non-Qwen models.

---

### Problem 2: Missing spatialreal Task Handler

**File**: `evals/evaluation.py`

**Symptom**: The evaluation script only had handlers for `spatialmap`, `mazenav`, and `spatialgrid`. The `spatialreal` task (135 samples) was silently dropped during evaluation, making accuracy scores artificially high.

**Fix**: Added `extract_answer_from_text_spatialreal()` function and the corresponding `elif task == 'spatialreal'` branch in `evaluate_model_accuracy()`.

**Impact**: Without this fix, 135 out of 4635 samples (2.9%) were excluded from evaluation.

---

### Problem 3: Flash Attention API Break (LLaVA-OV Models)

**Symptom**: LLaVA-OneVision-1.5 custom code imports `flash_attn_varlen_func` from `transformers.modeling_flash_attention_utils`, but this symbol was removed in Transformers 4.57.1. The actual function lives in the `flash_attn` package.

**Initial workaround** (suboptimal): Set `attn_implementation="eager"` which disabled flash attention entirely. This caused LLaVA-OV-8B to achieve near-0% accuracy.

**Correct fix**: Monkey-patch the missing symbol at import time:
```python
import transformers.modeling_flash_attention_utils as _fau
if not hasattr(_fau, 'flash_attn_varlen_func'):
    from flash_attn import flash_attn_varlen_func as _favf
    _fau.flash_attn_varlen_func = _favf
```

**Impact**: With eager attention, LLaVA-OV-8B scored ~0% on spatialreal. With the correct flash attention fix, it achieved 34.8% - the best among all models on that task.

---

### Problem 4: LLaVA-OV Cannot Use Batched Inference

**Symptom**: When `batch_size > 1`, LLaVA-OV custom attention code raises `AttributeError: 'MistralAttention' object has no attribute 'num_heads'` during the custom attention forward pass.

**Workaround**: Force `batch_size=1` for LLaVA-OV models. This reduces throughput but is the only reliable approach given their custom attention implementation.

**Root cause**: LLaVA-OV-1.5 uses custom modeling code (`trust_remote_code=True`) that assumes single-sample processing in its attention modules.

---

### Problem 5: pad_token_id Setting Breaks LLaVA-OV Model

**Symptom**: Setting `processor.tokenizer.pad_token = processor.tokenizer.eos_token` triggers a full model re-initialization for LLaVA-OV, causing it to re-download/re-process the model config and breaking the custom model weights.

**Fix**: Only set `padding_side = "left"` on the tokenizer, do NOT modify pad_token:
```python
# Safe:
if hasattr(processor, 'tokenizer'):
    processor.tokenizer.padding_side = "left"

# Dangerous (breaks LLaVA-OV):
# processor.tokenizer.pad_token = processor.tokenizer.eos_token
```

---

### Problem 6: MiniCPM-V Uses Non-Standard Chat API

**Symptom**: MiniCPM-V does not use the standard `model.generate()` pattern. Instead, it uses `model.chat(image, msgs, tokenizer, processor, ...)`. Additionally, the `chat()` method expects the raw tokenizer, not the processor, as the `tokenizer` argument.

**Fix**: Separate inference path for MiniCPM-V:
```python
def generate_minicpmv(model, processor, question, image, args):
    tokenizer = processor.tokenizer  # Extract actual tokenizer
    answer = model.chat(
        image=image,
        msgs=deepcopy(msgs),  # Must deepcopy
        tokenizer=tokenizer,
        processor=processor,
        max_new_tokens=args.max_new_tokens,
        sampling=args.temperature > 1e-5,
        temperature=args.temperature,
    )
```

---

### Problem 7: Corrupted HuggingFace Cached Modules

**Symptom**: Molmo2 loading fails with `KeyError: 'default'` in `ROPE_INIT_FUNCTIONS` after a previous model load modifies the global transformers state.

**Fix**: Clear the corrupted cache:
```bash
rm -rf /root/.cache/huggingface/modules/transformers_modules/
```

**Prevention**: Load models sequentially on the same GPU (our dual-GPU approach naturally avoids this since each GPU loads independently).

---

### Problem 8: Question Grouping Bias with `--task all`

**Symptom**: When using `--first_k` to sample N questions per group, the original code used `id.split('.')[-1]` as the group key, which grouped questions across different tasks together. This meant `first_k=10` would sample 10 items from mazenav.qid0 instead of sampling equally across all tasks.

**Fix**: Include task prefix in the group key:
```python
# Before:
group_key = item["id"].split(".")[-1]  # Groups all tasks together

# After:
task = item["id"].split(".")[0]
question_id = item["id"].split(".")[-1]
group_key = f"{task}.{question_id}"  # Separates tasks correctly
```

---

### Problem 9: Inode Exhaustion on /autodl-fs

**Symptom**: `/autodl-fs` filesystem has a 200K inode limit. Other datasets and model caches consumed most inodes, causing "No space left on device" errors even with free disk space.

**Fix**: Cleaned up `.locks` directories and other unnecessary files:
```bash
find /root/autodl-fs -name "*.lock" -delete
find /root/autodl-fs -name ".locks" -type d -exec rm -rf {} +
```

---

### Problem 10: Model Loading with device_map

**Symptom**: Using `device_map="auto"` can spread models across both GPUs, interfering with dual-GPU parallel execution.

**Fix**: Use `device_map={"": 0}` combined with `CUDA_VISIBLE_DEVICES`:
```bash
# GPU 0 sees only physical GPU 0, maps to cuda:0
CUDA_VISIBLE_DEVICES=0 python inference_unified.py --model_path ...

# GPU 1 sees only physical GPU 1, maps to cuda:0
CUDA_VISIBLE_DEVICES=1 python inference_unified.py --model_path ...
```

This ensures each process uses exactly one GPU without cross-GPU spreading.

---

## 5. Final Results Summary (11 Models)

| Rank | Model | Size | spatialmap | mazenav | spatialgrid | spatialreal | Overall |
|------|-------|------|-----------|---------|------------|------------|---------|
| 1 | Molmo2-8B | 8B | 73.7% | 37.3% | **88.1%** | 16.3% | **64.9%** |
| 2 | Qwen3-VL-4B | 4B | 69.0% | **46.3%** | 77.9% | 8.9% | **62.8%** |
| 3 | MiniCPM-V-4.5 | 4.5B | 69.6% | 46.1% | 77.1% | 13.3% | **62.8%** |
| 4 | Molmo2-4B | 4B | 67.5% | 37.5% | 85.8% | **17.8%** | **62.3%** |
| 5 | SAIL-VL2-8B | 8B | 71.7% | 30.5% | 87.3% | 12.6% | **61.7%** |
| 6 | Qwen3-VL-8B | 8B | **77.3%** | 33.0% | 76.3% | 10.4% | **60.7%** |
| 7 | LLaVA-OV-1.5-4B | 4B | 72.5% | 29.8% | 77.9% | 15.6% | **58.8%** |
| 8 | InternVL3-4B | 4B | 64.6% | 31.9% | 80.4% | 13.3% | **57.6%** |
| 9 | InternVL3-8B | 8B | 65.9% | 30.9% | 78.1% | 8.1% | **56.9%** |
| 10 | LLaVA-OV-1.5-8B | 8B | 68.0% | 18.7% | 81.5% | 34.8% | **55.5%** |
| 11 | Gemma-3-4b | 4B | 39.2% | 31.3% | 67.1% | 6.7% | **44.7%** |

---

## 6. Lessons Learned

1. **Always verify model-specific attention implementations** when upgrading transformers. Breaking changes in internal APIs can silently degrade performance.

2. **Batched inference provides significant speedup** (2.7x at bs=8) but not all custom model code supports it. Always test with batch_size=1 first.

3. **Dual-GPU parallel execution** via `CUDA_VISIBLE_DEVICES` + `device_map={"": 0}` is the simplest and most reliable approach for parallel model evaluation.

4. **Evaluation scripts can silently drop data**. The missing spatialreal handler meant 135 samples were invisible in accuracy calculations. Always verify that evaluation counts match dataset counts.

5. **Python truthiness bugs are dangerous** in ML pipelines. `"x" or "y" in string` always evaluates to `True` and can silently route inference through wrong code paths.

---

## 7. Additional Problems (5 New Models)

### Problem 11: SAIL-VL2 Config Rejects Default Architecture

**File**: SAIL-VL2 snapshot `configuration_sailvl.py`

**Symptom**: The config's `__init__` defaults `llm_config` to `{'architectures': ['Qwen2ForCausalLM']}`, but only accepts `Qwen3ForCausalLM` and `LlamaForCausalLM`. When transformers calls `__repr__()` during config loading, it creates a default instance that triggers `ValueError: Unsupported architecture`.

**Fix**: Changed the `else` branch to use `Qwen3Config` as fallback for all Qwen-family architectures.

---

### Problem 12: SAIL-VL2 Imports `LossKwargs` Missing from Transformers 4.57

**Symptom**: SAIL-VL2's custom `modeling_qwen3.py` imports `LossKwargs` from `transformers.utils`, which doesn't exist in transformers 4.57.1. This was added in a later transformers release.

**Fix**: Patched `modeling_qwen3.py` to remove the `LossKwargs` import and simplify `KwargsForCausalLM` to not inherit from it.

---

### Problem 13: Gemma-3 Processor Requires Nested Image Lists for Batching

**Symptom**: Gemma-3's processor expects `images=[[img1], [img2], ...]` (nested list) for batched inputs, not `images=[img1, img2, ...]` (flat list) like other models. Passing a flat list raises `ValueError: Received inconsistently sized batches`. The error was caught by the batch try/except, but the resulting generation produced **all empty answers** silently.

**Fix**: Added fallback in `generate_batch()` that catches the ValueError and retries with nested images.

**Impact**: First run produced 4635 empty answers (0% accuracy). Fixed run achieved 44.7% accuracy.

---

### Problem 14: InternVL3-8B Uses Different API Than InternVL3_5-4B

**Symptom**: InternVL3-8B uses `InternVLChatModel` (custom code, `model.chat()` API) while InternVL3_5-4B uses `InternVLForImageTextToText` (native transformers, `model.generate()`). Same model family, different inference APIs.

**Fix**: Added `internvl_chat` model type with InternVL-style dynamic image preprocessing and `model.chat()` wrapper.

---

### Problem 15: Batched Inference Not Possible for InternVL-Chat and SAIL-VL Models

**Symptom**: The `model.chat()` API in InternVL3-8B and SAIL-VL2 takes a single `pixel_values` tensor per call, with no batch dimension. This forces `batch_size=1`, making these models ~7x slower than batchable models.

**Workaround**: No alternative - these models must run at bs=1. One 8B model takes ~7 hours for 4635 samples.
