#!/bin/bash
# run_all_evals.sh - Run SpatialEval benchmark on all available VLMs.
#
# Usage:
#   bash scripts/run_all_evals.sh [MODE] [TASK]
#   MODE: vqa|vtqa|tqa (default: vqa)
#   TASK: all|spatialmap|mazenav|spatialgrid|spatialreal (default: all)
#
# Environment:
#   HF_HUB_CACHE=/root/autodl-fs/models
#   HF_DATASETS_CACHE=/root/autodl-fs/datasets

set -euo pipefail

export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/autodl-fs/models}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/autodl-fs/datasets}"

MODE="${1:-vqa}"
TASK="${2:-all}"
DEVICE="cuda"
OUTPUT_FOLDER="outputs"
MAX_NEW_TOKENS=512
TEMPERATURE=0.2

# Models to evaluate (HuggingFace IDs matching the cache)
MODELS=(
    "OpenGVLab/InternVL3_5-4B-HF"
    "Qwen/Qwen3-VL-4B-Instruct"
    "allenai/Molmo2-4B"
    "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    "openbmb/MiniCPM-V-4_5"
)

echo "=============================================="
echo "SpatialEval Benchmark Runner"
echo "=============================================="
echo "Mode: ${MODE}"
echo "Task: ${TASK}"
echo "HF_HUB_CACHE: ${HF_HUB_CACHE}"
echo "HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"
echo "Models: ${#MODELS[@]}"
echo "=============================================="

FAILED=()
SUCCEEDED=()

for MODEL in "${MODELS[@]}"; do
    MODEL_TAG=$(echo "${MODEL}" | tr '/' '__')
    echo ""
    echo ">>> [$(date '+%H:%M:%S')] Running: ${MODEL} (${MODE}/${TASK})"

    LOG_FILE="outputs/logs/${MODEL_TAG}_${MODE}_${TASK}_w_reason.log"
    mkdir -p "$(dirname "${LOG_FILE}")"

    if python inference_unified.py \
        --model_path "${MODEL}" \
        --mode "${MODE}" \
        --task "${TASK}" \
        --w_reason \
        --output_folder "${OUTPUT_FOLDER}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --temperature "${TEMPERATURE}" \
        --device "${DEVICE}" \
        2>&1 | tee "${LOG_FILE}"; then
        SUCCEEDED+=("${MODEL}")
        echo ">>> SUCCESS: ${MODEL}"
    else
        FAILED+=("${MODEL}")
        echo ">>> FAILED: ${MODEL} (see ${LOG_FILE})"
    fi
done

echo ""
echo "=============================================="
echo "Summary"
echo "=============================================="
echo "Succeeded: ${#SUCCEEDED[@]}"
for m in "${SUCCEEDED[@]:-}"; do echo "  + ${m}"; done
echo "Failed: ${#FAILED[@]}"
for m in "${FAILED[@]:-}"; do echo "  - ${m}"; done
echo "=============================================="

# Run evaluation if any succeeded
if [ ${#SUCCEEDED[@]} -gt 0 ]; then
    echo ""
    echo "Running evaluation..."
    for TASK_NAME in spatialmap mazenav spatialgrid; do
        if [ "${TASK}" = "all" ] || [ "${TASK}" = "${TASK_NAME}" ]; then
            mkdir -p "eval_summary/${MODE}"
            python evals/evaluation.py \
                --mode "${MODE}" \
                --task "${TASK_NAME}" \
                --output_folder "${OUTPUT_FOLDER}" \
                --dataset_id "MilaWang/SpatialEval" \
                --eval_summary_dir "eval_summary" 2>&1 || true
        fi
    done
fi
