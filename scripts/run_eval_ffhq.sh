#!/bin/bash
# Evaluate restoration results on FFHQ (Tables 1-2).
# Usage: bash scripts/run_eval_ffhq.sh [num_samples]

export CUDA_VISIBLE_DEVICES=0
NUM_SAMPLES="${1:-150}"

for TASK in random_inpainting super_resolution gaussian_blur motion_blur; do
    for METHOD in dps mpgd dsg admmdiff diffrgd; do
        uv run evaluation/evaluate_restoration.py -t "$TASK" -m "$METHOD" -d ffhq -n "$NUM_SAMPLES"
    done
done
