#!/bin/bash
# Download the raw OpenAI guided-diffusion checkpoints into checkpoints/temp/,
# then convert them with convert_openai_model_to_diffuser.py.

mkdir -p checkpoints/temp
cd checkpoints/temp

gdown 1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh  # FFHQ (ffhq_10m.pt, from DPS)
gdown 1HAy7P19PckQLczVNXmVF-e_CRxq098uW  # ImageNet (imagenet256.pt, from Dhariwal & Nichol)
