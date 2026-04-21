#!/bin/bash
# run_full_batched.sh - Full SpatialEval VQA evaluation with batching.
# Uses both GPUs in parallel: GPU 0 runs models 1-3, GPU 1 runs models 4-6.
set -euo pipefail

export HF_HUB_CACHE=/root/autodl-fs/models
export HF_DATASETS_CACHE=/root/autodl-fs/datasets

echo "=== Full SpatialEval VQA Evaluation (batch_size=8, 4635 samples/model) ==="
echo "Started at: $(date)"

# GPU 0 models
GPU0_MODELS=(
    "Qwen/Qwen3-VL-4B-Instruct"
    "OpenGVLab/InternVL3_5-4B-HF"
    "allenai/Molmo2-4B"
)

# GPU 1 models
GPU1_MODELS=(
    "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    "openbmb/MiniCPM-V-4_5"
)

run_model() {
    local MODEL="$1"
    local GPU="$2"
    local BS="$3"
    local MINICPM="false"
    [[ "$MODEL" == *"MiniCPM"* ]] && MINICPM="true"

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
    for MODEL in "${GPU0_MODELS[@]}"; do
        BS=8
        [[ "$MODEL" == *"MiniCPM"* ]] && BS=1
        run_model "$MODEL" 0 $BS
    done
    echo "=== GPU 0 DONE at $(date) ==="
) &

GPU0_PID=$!

# Launch GPU 1 pipeline in background
(
    for MODEL in "${GPU1_MODELS[@]}"; do
        BS=8
        [[ "$MODEL" == *"MiniCPM"* ]] && BS=1
        run_model "$MODEL" 1 $BS
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
