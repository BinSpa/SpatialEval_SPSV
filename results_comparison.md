# SpatialEval Benchmark - VLM Comparison Report

**Date**: 2025-04-21
**Mode**: VQA (Vision Question Answering) with chain-of-thought reasoning (`--w_reason`)
**Samples**: 4635 per model (full dataset, no sampling)
**Hardware**: NVIDIA A800 80GB (x2), dual-GPU parallel inference
**Max new tokens**: 128, Temperature: 0.2

## Final Results (11 models, 4635 samples/model)

| Rank | Model | Size | spatialmap | mazenav | spatialgrid | spatialreal | **Overall** |
|------|-------|------|-----------|---------|------------|------------|-------------|
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

## Per-Task Leaders

| Task | Best Model | Accuracy | Description |
|------|-----------|----------|-------------|
| spatialmap | Qwen3-VL-8B | **77.3%** | Direction/position on maps |
| mazenav | Qwen3-VL-4B | **46.3%** | Maze path counting |
| spatialgrid | Molmo2-8B | **88.1%** | Grid counting/identification |
| spatialreal | LLaVA-OV-1.5-8B | **34.8%** | Real-world spatial reasoning |

## Key Findings

### Scaling (4B vs 8B within same family)

| Family | 4B | 8B | Delta |
|--------|-----|-----|-------|
| Molmo2 | 62.3% | **64.9%** | +2.6pp |
| Qwen3-VL | 62.8% | 60.7% | -2.1pp |
| InternVL3 | 57.6% | 56.9% | -0.7pp |
| LLaVA-OV-1.5 | 58.8% | 55.5% | -3.3pp |

- Only **Molmo2** benefits from scaling up (4B → 8B: +2.6pp)
- Qwen3-VL, InternVL3, and LLaVA-OV all **degrade** at 8B scale
- LLaVA-OV-8B's overall drop masks a massive spatialreal gain (+19.2pp)

### Model Rankings by Tier

1. **Top tier** (62-65%): Molmo2-8B, Qwen3-VL-4B, MiniCPM-V-4.5, Molmo2-4B, SAIL-VL2-8B
2. **Mid tier** (56-61%): Qwen3-VL-8B, LLaVA-OV-4B, InternVL3-4B, InternVL3-8B
3. **Lower tier** (44-56%): LLaVA-OV-8B (best spatialreal but poor elsewhere), Gemma-3-4b

### Notable Observations

- **Molmo2-8B** is the overall winner, leading in spatialgrid (88.1%) and consistent across all tasks
- **Qwen3-VL-4B** dominates mazenav (46.3%) - best at maze reasoning
- **LLaVA-OV-1.5-8B** is paradoxical: worst overall (55.5%) but best at spatialreal (34.8%) by a huge margin
- **Gemma-3-4b** ranks last (44.7%), significantly behind other 4B models
- **SAIL-VL2-8B** (61.7%) performs competitively for a newer model, strong at spatialgrid (87.3%)

## Output Files

All located at `outputs/MilaWang__SpatialEval/vqa/all/` (11 files, 4635 lines each):
- `m-Qwen__Qwen3-VL-4B-Instruct_w_reason.jsonl`
- `m-Qwen__Qwen3-VL-8B-Instruct_w_reason.jsonl`
- `m-OpenGVLab__InternVL3_5-4B-HF_w_reason.jsonl`
- `m-OpenGVLab__InternVL3-8B-Instruct_w_reason.jsonl`
- `m-allenai__Molmo2-4B_w_reason.jsonl`
- `m-allenai__Molmo2-8B_w_reason.jsonl`
- `m-lmms-lab__LLaVA-OneVision-1.5-4B-Instruct_w_reason.jsonl`
- `m-lmms-lab__LLaVA-OneVision-1.5-8B-Instruct_w_reason.jsonl`
- `m-openbmb__MiniCPM-V-4_5_w_reason.jsonl`
- `m-BytedanceDouyinContent__SAIL-VL2-8B_w_reason.jsonl`
- `m-google__gemma-3-4b-it_w_reason.jsonl`

## Files Created/Modified

- `inference_unified.py` - Unified VLM inference supporting 11 models (6 architectures)
- `evals/evaluation.py` - Added spatialreal task handler
- `inference_vlm.py` - Fixed Python truthiness bug on line 139
- `scripts/run_full_eval.sh` - Dual-GPU evaluation runner (6 original models)
- `scripts/run_new_models.sh` - Dual-GPU evaluation runner (5 new models)
- `scripts/compare_results.py` - Results comparison tool
- `results_comparison.md` - This report
- `PROJECT_REPORT.md` - Comprehensive project documentation
