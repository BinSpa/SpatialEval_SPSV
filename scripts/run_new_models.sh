#!/bin/bash
# run_new_models.sh - Evaluate 5 new models using both GPUs in parallel.
set -euo pipefail

export HF_HUB_CACHE=/root/autodl-fs/models
export HF_DATASETS_CACHE=/root/autodl-fs/datasets

echo "=== Evaluating 5 new models (full 4635 samples/model) ==="
echo "Started at: $(date)"

# GPU 0: Standard generate models (batchable)
GPU0_MODELS=(
    "Qwen/Qwen3-VL-8B-Instruct:8"
    "allenai/Molmo2-8B:8"
    "google/gemma-3-4b-it:8"
)

# GPU 1: Custom chat API models (bs=1)
GPU1_MODELS=(
    "OpenGVLab/InternVL3-8B-Instruct:1"
    "BytedanceDouyinContent/SAIL-VL2-8B:1"
)

run_model() {
    local MODEL="$1"
    local GPU="$2"
    local BS="$3"

    echo ">>> [$(date '+%H:%M:%S')] GPU${GPU} Starting: ${MODEL} (bs=${BS})"
    if CUDA_VISIBLE_DEVICES=${GPU} python -u inference_unified.py \
        --model_path "${MODEL}" \
        --mode vqa --task all --w_reason \
        --batch_size ${BS} \
        --output_folder outputs \
        --max_new_tokens 128 \
        --temperature 0.2 \
        --device cuda; then
        echo ">>> [$(date '+%H:%M:%S')] GPU${GPU} SUCCESS: ${MODEL}"
    else
        echo ">>> [$(date '+%H:%M:%S')] GPU${GPU} FAILED: ${MODEL}"
    fi
}

# Launch GPU 0 pipeline in background
(
    for entry in "${GPU0_MODELS[@]}"; do
        MODEL="${entry%%:*}"
        BS="${entry##*:}"
        run_model "$MODEL" 0 "$BS"
    done
    echo "=== GPU 0 DONE at $(date) ==="
) &
GPU0_PID=$!

# Launch GPU 1 pipeline in background
(
    for entry in "${GPU1_MODELS[@]}"; do
        MODEL="${entry%%:*}"
        BS="${entry##*:}"
        run_model "$MODEL" 1 "$BS"
    done
    echo "=== GPU 1 DONE at $(date) ==="
) &
GPU1_PID=$!

echo "GPU 0 pipeline PID: ${GPU0_PID}"
echo "GPU 1 pipeline PID: ${GPU1_PID}"
echo "Waiting for both pipelines to complete..."

wait ${GPU0_PID}
wait ${GPU1_PID}

echo ""
echo "=== ALL DONE at $(date) ==="
