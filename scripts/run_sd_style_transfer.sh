#!/bin/bash
# Style-guided generation with Stable Diffusion v1.5 (Table 10, Appendix E.2).

export CUDA_VISIBLE_DEVICES=0

uv run run_sd_style_transfer.py -m dps     --step_size 2
uv run run_sd_style_transfer.py -m mpgd    --step_size 150
uv run run_sd_style_transfer.py -m dsg     --step_size 0.1
uv run run_sd_style_transfer.py -m diffrgd --step_size 2.5
