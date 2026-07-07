#!/bin/bash
# Image restoration on FFHQ with 1000 DDIM steps (Table 1).
# Usage: bash scripts/run_ffhq_t1000.sh [num_samples]

export CUDA_VISIBLE_DEVICES=0
NUM_SAMPLES="${1:-150}"

# DPS
uv run run_inverse_problem.py -t random_inpainting -m dps -d ffhq --step_size 1  --eta 1 --num_inference_steps 1000 --noise gaussian -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t super_resolution  -m dps -d ffhq --step_size 1  --eta 1 --num_inference_steps 1000 --noise gaussian -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t gaussian_blur     -m dps -d ffhq --step_size 1  --eta 1 --num_inference_steps 1000 --noise gaussian -n "$NUM_SAMPLES" 
uv run run_inverse_problem.py -t motion_blur       -m dps -d ffhq --step_size 1  --eta 1 --num_inference_steps 1000 --noise gaussian -n "$NUM_SAMPLES"

# MPGD
uv run run_inverse_problem.py -t random_inpainting -m mpgd -d ffhq --step_size 30  --eta 1 --num_inference_steps 1000 --noise gaussian -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t super_resolution  -m mpgd -d ffhq --step_size 30  --eta 1 --num_inference_steps 1000 --noise gaussian -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t gaussian_blur     -m mpgd -d ffhq --step_size 50  --eta 1 --num_inference_steps 1000 --noise gaussian -n "$NUM_SAMPLES" 
uv run run_inverse_problem.py -t motion_blur       -m mpgd -d ffhq --step_size 50  --eta 1 --num_inference_steps 1000 --noise gaussian -n "$NUM_SAMPLES" 

# DSG
uv run run_inverse_problem.py -t random_inpainting -m dsg -d ffhq --step_size 0.2 -i 5  --eta 1 --num_inference_steps 1000 --noise gaussian -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t super_resolution  -m dsg -d ffhq --step_size 0.2 -i 20 --eta 1 --num_inference_steps 1000 --noise gaussian -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t gaussian_blur     -m dsg -d ffhq --step_size 0.2 -i 5  --eta 1 --num_inference_steps 1000 --noise gaussian -n "$NUM_SAMPLES" 
uv run run_inverse_problem.py -t motion_blur       -m dsg -d ffhq --step_size 0.2 -i 5  --eta 1 --num_inference_steps 1000 --noise gaussian -n "$NUM_SAMPLES"

# ADMMDiff
uv run run_inverse_problem.py -t random_inpainting -m admmdiff -d ffhq --step_size 1 --num_inference_steps 1000 --noise gaussian --inner_max_iter 1 --rho 0.9 -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t super_resolution  -m admmdiff -d ffhq --step_size 1 --num_inference_steps 1000 --noise gaussian --inner_max_iter 1 --rho 0.9 -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t gaussian_blur     -m admmdiff -d ffhq --step_size 1 --num_inference_steps 1000 --noise gaussian --inner_max_iter 1 --rho 0.9 -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t motion_blur       -m admmdiff -d ffhq --step_size 1 --num_inference_steps 1000 --noise gaussian --inner_max_iter 1 --rho 0.9 -n "$NUM_SAMPLES"

# DiffRGD
uv run run_inverse_problem.py -t random_inpainting -m diffrgd -d ffhq --step_size 5   -i 5  --eta 1 --num_inference_steps 1000 --noise gaussian --inner_max_iter 3 -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t super_resolution  -m diffrgd -d ffhq --step_size 5   -i 10 --eta 1 --num_inference_steps 1000 --noise gaussian --inner_max_iter 3 -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t gaussian_blur     -m diffrgd -d ffhq --step_size 5   -i 5  --eta 1 --num_inference_steps 1000 --noise gaussian --inner_max_iter 3 -n "$NUM_SAMPLES"
uv run run_inverse_problem.py -t motion_blur       -m diffrgd -d ffhq --step_size 2.5 -i 4  --eta 1 --num_inference_steps 1000 --noise gaussian --inner_max_iter 3 -n "$NUM_SAMPLES"
