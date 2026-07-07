#!/bin/bash
# Image restoration on ImageNet with 100 DDIM steps (Table 3).
# Usage: bash scripts/run_imagenet_t100.sh [num_samples]

export CUDA_VISIBLE_DEVICES=0
NUM_SAMPLES="${1:-150}"
NUM_STEP=100

# DPS
uv run run_inverse_problem.py -t random_inpainting -m dps -d imagenet --step_size 1    --eta 1 --num_inference_steps "$NUM_STEP" --noise gaussian -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t super_resolution  -m dps -d imagenet --step_size 1    --eta 1 --num_inference_steps "$NUM_STEP" --noise gaussian -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t gaussian_blur     -m dps -d imagenet --step_size 0.4  --eta 1 --num_inference_steps "$NUM_STEP" --noise gaussian -n "$NUM_SAMPLES" 
uv run run_inverse_problem.py -t motion_blur       -m dps -d imagenet --step_size 0.6  --eta 1 --num_inference_steps "$NUM_STEP" --noise gaussian -n "$NUM_SAMPLES"

# MPGD
uv run run_inverse_problem.py -t random_inpainting -m mpgd -d imagenet --step_size 20  --eta 1 --num_inference_steps 100 --noise gaussian -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t super_resolution  -m mpgd -d imagenet --step_size 25  --eta 1 --num_inference_steps 100 --noise gaussian -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t gaussian_blur     -m mpgd -d imagenet --step_size 10  --eta 1 --num_inference_steps 100 --noise gaussian -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t motion_blur       -m mpgd -d imagenet --step_size 10  --eta 1 --num_inference_steps 100 --noise gaussian -n "$NUM_SAMPLES"

# DSG
uv run run_inverse_problem.py -t random_inpainting -m dsg -d imagenet --step_size 0.2  --eta 1 --num_inference_steps "$NUM_STEP" --noise gaussian -i 5  -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t super_resolution  -m dsg -d imagenet --step_size 0.1  --eta 1 --num_inference_steps "$NUM_STEP" --noise gaussian -i 10 -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t gaussian_blur     -m dsg -d imagenet --step_size 0.1  --eta 1 --num_inference_steps "$NUM_STEP" --noise gaussian -i 5  -n "$NUM_SAMPLES" 
uv run run_inverse_problem.py -t motion_blur       -m dsg -d imagenet --step_size 0.1  --eta 1 --num_inference_steps "$NUM_STEP" --noise gaussian -i 5  -n "$NUM_SAMPLES"

# ADMMDiff
uv run run_inverse_problem.py -t random_inpainting -m admmdiff -d imagenet --step_size 0.8 --rho 0.4 --num_inference_steps "$NUM_STEP" --noise gaussian --inner_max_iter 10 -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t super_resolution  -m admmdiff -d imagenet --step_size 0.8 --rho 0.4 --num_inference_steps "$NUM_STEP" --noise gaussian --inner_max_iter 10 -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t gaussian_blur     -m admmdiff -d imagenet --step_size 2.4 --num_inference_steps "$NUM_STEP" --noise gaussian --inner_max_iter 10 -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t motion_blur       -m admmdiff -d imagenet --step_size 2.4 --num_inference_steps "$NUM_STEP" --noise gaussian --inner_max_iter 10 -n "$NUM_SAMPLES"

# DiffRGD
uv run run_inverse_problem.py -t random_inpainting -m diffrgd -d imagenet --step_size 5   --eta 1 --num_inference_steps "$NUM_STEP" --noise gaussian -i 1 --inner_max_iter 3 -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t super_resolution  -m diffrgd -d imagenet --step_size 5   --eta 1 --num_inference_steps "$NUM_STEP" --noise gaussian -i 1 --inner_max_iter 3 -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t gaussian_blur     -m diffrgd -d imagenet --step_size 0.6 --eta 1 --num_inference_steps "$NUM_STEP" --noise gaussian -i 1 --inner_max_iter 3 -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t motion_blur       -m diffrgd -d imagenet --step_size 0.5 --eta 1 --num_inference_steps "$NUM_STEP" --noise gaussian -i 1 --inner_max_iter 3 -n "$NUM_SAMPLES"
