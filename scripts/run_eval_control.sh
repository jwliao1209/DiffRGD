#!/bin/bash
# Evaluate conditional generation results (Table 5).
# Usage: bash scripts/run_eval_control.sh [num_samples]

NUM_SAMPLES="${1:-150}"

for TASK in segmentation sketch face_id; do
    for METHOD in dps dsg admmdiff diffrgd; do
        uv run evaluation/evaluate_control.py -t "$TASK" -m "$METHOD" -n "$NUM_SAMPLES"
    done
done
