#!/bin/bash
# Generate measurement datasets for all restoration tasks.
# Usage: bash scripts/generate_dataset.sh [num_samples] [dataset]

NUM_SAMPLES="${1:-150}"
DATASET="${2:-ffhq}"

uv run generate_dataset.py --task random_inpainting --dataset "$DATASET" --num_samples "$NUM_SAMPLES"
uv run generate_dataset.py --task box_inpainting    --dataset "$DATASET" --num_samples "$NUM_SAMPLES"
uv run generate_dataset.py --task super_resolution  --dataset "$DATASET" --num_samples "$NUM_SAMPLES"
uv run generate_dataset.py --task gaussian_blur     --dataset "$DATASET" --num_samples "$NUM_SAMPLES"
uv run generate_dataset.py --task motion_blur       --dataset "$DATASET" --num_samples "$NUM_SAMPLES"

# Denoising datasets at different noise levels (Appendix E.1)
uv run generate_dataset.py --task denoising --dataset "$DATASET" --num_samples "$NUM_SAMPLES" --gaussian_noise_level 0.1
uv run generate_dataset.py --task denoising --dataset "$DATASET" --num_samples "$NUM_SAMPLES" --gaussian_noise_level 0.2
uv run generate_dataset.py --task denoising --dataset "$DATASET" --num_samples "$NUM_SAMPLES" --gaussian_noise_level 0.3
