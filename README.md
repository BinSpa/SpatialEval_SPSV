# SpatialEval - Large-Scale VLM Evaluation

Evaluation of **11 Vision-Language Models** on the [SpatialEval benchmark](https://arxiv.org/abs/2406.14852) (NeurIPS 2024), a spatial reasoning benchmark with 4,635 samples per mode across 4 tasks.

## Results Overview (VQA mode, full 4635 samples/model)

| Rank | Model | Size | spatialmap | mazenav | spatialgrid | spatialreal | Overall |
|------|-------|------|-----------|---------|------------|------------|---------|
| 1 | **Molmo2-8B** | 8B | 73.7% | 37.3% | **88.1%** | 16.3% | **64.9%** |
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

> Full analysis in [results_comparison.md](results_comparison.md) and [PROJECT_REPORT.md](PROJECT_REPORT.md).

## Benchmark Structure

SpatialEval tests spatial reasoning across 4 tasks:

| Task | Description | Samples | Question Types |
|------|-------------|---------|----------------|
| spatialmap | Direction/position on synthetic maps | 1,500 | direction, nearest object, count |
| mazenav | Path counting/navigation in mazes | 1,500 | path count, step count, reachability |
| spatialgrid | Animal counting/identification in 5x5 grids | 1,500 | count, identify, identify at position |
| spatialreal | Real-world image spatial reasoning | 135 | varied counting/identification |

Each task supports 3 modes: TQA (text-only), VQA (vision), VTQA (vision+text). Total: 13,905 samples.

## Quick Start

```bash
export HF_HUB_CACHE=/path/to/models
export HF_DATASETS_CACHE=/path/to/datasets

# Run a single model
python inference_unified.py \
    --model_path "Qwen/Qwen3-VL-4B-Instruct" \
    --mode vqa --task all --w_reason \
    --batch_size 8 \
    --output_folder outputs --device cuda

# Evaluate all models on both GPUs
bash scripts/run_full_eval.sh
```

### Supported Models

`inference_unified.py` auto-detects and supports 11 models across 6 architectures:

- **InternVL** (InternVL3_5-4B, InternVL3-8B) - native and custom chat API
- **Qwen3-VL** (4B, 8B) - native AutoModelForImageTextToText
- **Molmo2** (4B, 8B) - native AutoModelForImageTextToText
- **LLaVA-OV-1.5** (4B, 8B) - custom code with flash attention patch
- **MiniCPM-V-4.5** - custom model.chat() API
- **SAIL-VL2-8B** - InternVL-style with Qwen3 backbone
- **Gemma-3-4b** - native AutoModelForImageTextToText

### Evaluation

```bash
python evals/evaluation.py --mode vqa --task spatialmap --output_folder outputs/
```

## Output Format

Each model produces a JSONL file at `outputs/MilaWang__SpatialEval/vqa/all/`. Each line is one sample:

```json
{
  "id": "spatialmap.vqa.2000.0",
  "answer": "A. Northeast.\n\nStep 1: Locate \"Police Supply Store\"...",
  "oracle_answer": "Northeast",
  "oracle_option": "A",
  "oracle_full_answer": "Northeast (option A)",
  "prompt": "...",
  "image": "dataset"
}
```

Fields:
- `id` - Sample identifier in `{task}.{mode}.{instance_id}.{question_id}` format
- `answer` - Raw model output (free-form text with reasoning when using `--w_reason`)
- `oracle_answer` - Ground truth answer
- `oracle_option` - Correct option letter (A/B/C/D)
- `oracle_full_answer` - Full oracle answer with option
- `image` - Image source: `"dataset"`, `"random"`, `"noise"`, or `""` (text-only)

## How Accuracy is Calculated

Accuracy is computed in two steps:

### Step 1: Answer Extraction

Each task has a dedicated regex-based extraction function in `evals/evaluation.py` that parses the free-form model answer into a clean string:

| Task | Function | Extraction Strategy |
|------|----------|-------------------|
| spatialmap | `extract_answer_from_text_spatialmap()` | Direction words (qid0), object names (qid1), counts (qid2) |
| mazenav | `extract_answer_from_text_mazenav()` | Path/step counts (qid0/1), yes/no (qid2) |
| spatialgrid | `extract_answer_from_text_spatialgrid()` | Counts (qid0), animal names (qid1/2) |
| spatialreal | `extract_answer_from_text_spatialreal()` | Option letters, number words, or digits |

For example, given the raw answer `"A. Northeast.\n\nStep 1: ..."`, the spatialmap extractor (qid0) would extract `"northeast"`.

### Step 2: Substring Matching

The extracted answer is compared against the oracle answer using **substring matching** (line 350 of `evals/evaluation.py`):

```python
eval_result = int(ref_ans.lower() in model_answer.lower())
```

Accuracy = `correct_answers / total_samples`

### Known Limitation

The substring matching can produce **false positives**:
- Oracle `"six"` matches model answer `"sixteen"` (substring containment)
- Oracle `"2"` matches model answer `"12"`

This inflates accuracy scores, especially for counting tasks (mazenav qid0/qid1, spatialgrid qid0, spatialreal). A stricter exact-match or option-letter-only evaluation would likely yield different (lower) scores. See "Further Analysis Ideas > Answer Extraction Robustness" for improvement suggestions.

## What We Found

### Bugs Fixed in Original Codebase

1. **Python truthiness bug** (`inference_vlm.py:139`): `"qwen" or "cog"` always evaluates to `True`
2. **Missing spatialreal handler** (`evals/evaluation.py`): 135 samples silently dropped
3. **Question grouping bias**: `--first_k` sampled unequally across tasks

### Scaling Insights

| Family | 4B | 8B | Delta |
|--------|-----|-----|-------|
| Molmo2 | 62.3% | **64.9%** | +2.6pp |
| Qwen3-VL | 62.8% | 60.7% | -2.1pp |
| InternVL3 | 57.6% | 56.9% | -0.7pp |
| LLaVA-OV | 58.8% | 55.5% | -3.3pp |

Only Molmo2 benefits from scaling up. All other families degrade or stay flat at 8B.

### Task Difficulty

- **spatialgrid** (67-88%): Easiest - structured grid makes counting tractable
- **spatialmap** (39-77%): Moderate - requires understanding relative positions
- **mazenav** (19-46%): Hard - path counting demands multi-step reasoning
- **spatialreal** (7-35%): Hardest - real-world spatial reasoning remains a major challenge

## Further Analysis Ideas

### 1. Error Analysis by Question Type
Break down accuracy by qid within each task (e.g., direction vs. counting in spatialmap). This reveals whether models struggle with specific reasoning types (e.g., counting paths vs. determining reachability in mazenav).

### 2. Cross-Mode Comparison (TQA vs VQA vs VTQA)
Run TQA (text-only) and VTQA modes to measure how much vision actually helps. Compare VQA - TQA delta per model to quantify visual spatial grounding ability. Does adding an image improve or hurt performance?

### 3. Scaling Laws within Model Families
With 4B and 8B variants for Molmo2, Qwen3-VL, InternVL3, and LLaVA-OV, fit scaling curves. The surprising finding that most families degrade at 8B warrants investigation - is it architecture-specific (LLaVA-OV's custom attention), data issues, or something fundamental about spatial reasoning scaling?

### 4. Chain-of-Thought vs Direct Answer Ablation
Compare `--w_reason` vs `--bare` vs `--completion` prompts. Does reasoning help or hurt? Some models may produce correct short answers but incorrect reasoning, or vice versa.

### 5. Answer Extraction Robustness
The evaluation uses substring matching (`oracle_answer in model_answer`), which can cause false positives (e.g., "six" matching "sixteen"). Implement stricter exact-match or option-letter-only evaluation and compare scores. This affects absolute rankings but may also change relative rankings.

### 6. Per-Instance Difficulty Analysis
Identify which specific maze/grid/map instances are hardest across all models. Cluster by difficulty to understand what spatial configurations are universally challenging. Correlate with image complexity metrics (object count, path length, grid density).

### 7. Attention Mechanism Ablation
LLaVA-OV-8B dropped from expected ~60% to ~0% with eager attention vs 55.5% with flash attention. Systematically test eager vs flash attention for all models to quantify attention implementation effects on spatial reasoning.

### 8. Latency-Accuracy Tradeoff
Plot inference throughput (samples/min) vs accuracy for all 11 models. This Pareto analysis identifies the most efficient models for production deployment. Include batch size as a variable.

### 9. Model Answer Consistency
For questions with multiple instances of the same maze/grid/map (same image, same question, different random variations), measure answer consistency. High accuracy with low consistency suggests memorization rather than understanding.

### 10. Correlation with General VLM Benchmarks
Compare SpatialEval scores with general benchmarks (MMMU, MMMU-Pro, MathVista, AI2D). Is spatial reasoning a separate capability or correlated with general visual understanding?

## Citation

```
@inproceedings{wang2024spatial,
    title={Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models},
    author={Wang, Jiayu and Ming, Yifei and Shi, Zhenmei and Vineet, Vibhav and Wang, Xin and Li, Yixuan and Joshi, Neel},
    booktitle={The Thirty-Eighth Annual Conference on Neural Information Processing Systems},
    year={2024}
}
```
