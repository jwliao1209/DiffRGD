#!/bin/bash
# Evaluate restoration results on ImageNet (Table 3).
# Usage: bash scripts/run_eval_imagenet.sh [num_samples]

export CUDA_VISIBLE_DEVICES=0
NUM_SAMPLES="${1:-150}"

for TASK in random_inpainting super_resolution; do
    for METHOD in dps mpgd dsg admmdiff diffrgd; do
        uv run evaluation/evaluate_restoration.py -t "$TASK" -m "$METHOD" -d imagenet -n "$NUM_SAMPLES"
    done
done
