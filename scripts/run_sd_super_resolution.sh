#!/bin/bash
# Super-resolution with Stable Diffusion at 8x (512) and 12x (768) (Table 4).
# Usage: bash scripts/run_sd_super_resolution.sh [num_samples]

export CUDA_VISIBLE_DEVICES=0
NUM_SAMPLES="${1:-100}"

for RES in 512 768; do
    uv run run_inverse_problem.py -t super_resolution -m dps     -d ffhq --model ldm --resolution "$RES" --step_size 3   --eta 1 --num_inference_steps 250 --noise gaussian -n "$NUM_SAMPLES"
    uv run run_inverse_problem.py -t super_resolution -m dsg     -d ffhq --model ldm --resolution "$RES" --step_size 0.1 --eta 1 --num_inference_steps 250 --noise gaussian -i 2 -n "$NUM_SAMPLES"
    uv run run_inverse_problem.py -t super_resolution -m diffrgd -d ffhq --model ldm --resolution "$RES" --step_size 3.8 --eta 1 --num_inference_steps 250 --noise gaussian -i 1 --inner_max_iter 3 -n "$NUM_SAMPLES"
done
