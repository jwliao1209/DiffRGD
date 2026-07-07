#!/bin/bash
# Conditional generation on CelebA-HQ (Table 5).
# Usage: bash scripts/run_control.sh [num_samples]

export CUDA_VISIBLE_DEVICES=0
NUM_SAMPLES="${1:-150}"

# Segmentation map guidance
uv run run_control.py -t segmentation -m dps      --step_size 10  -n "$NUM_SAMPLES"
uv run run_control.py -t segmentation -m mpgd     --step_size 120 -n "$NUM_SAMPLES"
uv run run_control.py -t segmentation -m dsg      --step_size 0.1 -n "$NUM_SAMPLES"
uv run run_control.py -t segmentation -m admmdiff --step_size 5   -n "$NUM_SAMPLES" --inner_max_iter 10 --rho 0.1
uv run run_control.py -t segmentation -m diffrgd  --step_size 30  -n "$NUM_SAMPLES" --inner_max_iter 3

# Sketch guidance
uv run run_control.py -t sketch -m dps      --step_size 0.5  -n "$NUM_SAMPLES"
uv run run_control.py -t sketch -m mpgd     --step_size 10   -n "$NUM_SAMPLES"
uv run run_control.py -t sketch -m dsg      --step_size 0.08 -n "$NUM_SAMPLES"
uv run run_control.py -t sketch -m admmdiff --step_size 2    -n "$NUM_SAMPLES" --inner_max_iter 10 --rho 0.5
uv run run_control.py -t sketch -m diffrgd  --step_size 2    -n "$NUM_SAMPLES" --inner_max_iter 3

# FaceID guidance
uv run run_control.py -t face_id -m dps      --step_size 5     -n "$NUM_SAMPLES"
uv run run_control.py -t face_id -m dsg      --step_size 0.025 -n "$NUM_SAMPLES"
uv run run_control.py -t face_id -m admmdiff --step_size 3     -n "$NUM_SAMPLES" --inner_max_iter 10 --rho 0.2
uv run run_control.py -t face_id -m diffrgd  --step_size 8     -n "$NUM_SAMPLES" --inner_max_iter 3
