#!/bin/bash
# Image denoising on ImageNet at three noise levels (Table 9, Appendix E.1).
# Usage: bash scripts/run_denoising.sh [num_samples]

export CUDA_VISIBLE_DEVICES=0
NUM_SAMPLES="${1:-150}"

for SIGMA in 0.1 0.2 0.3; do
    uv run run_inverse_problem.py -t denoising_$SIGMA -m dps     -d imagenet --step_size 1    --eta 1 --num_inference_steps 100 --noise gaussian -n "$NUM_SAMPLES"
    uv run run_inverse_problem.py -t denoising_$SIGMA -m dsg     -d imagenet --step_size 0.15 --eta 1 --num_inference_steps 100 --noise gaussian -i 5 -n "$NUM_SAMPLES"
    uv run run_inverse_problem.py -t denoising_$SIGMA -m diffrgd -d imagenet --step_size 5    --eta 1 --num_inference_steps 100 --noise gaussian -i 1 --inner_max_iter 3 -n "$NUM_SAMPLES"
done
