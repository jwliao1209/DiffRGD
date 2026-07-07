<div align="center">

  <h1>
    DiffRGD: An Inference-Time Diffusion Guidance<br>
    Through Riemannian Gradient Descent
  </h1>

  <p>
    <a href="https://arxiv.org/abs/2606.28417">
      <img src="https://img.shields.io/badge/arXiv-2606.28417-b31b1b.svg" alt="arXiv" />
    </a>
    &nbsp;
    <a href="https://diffrgd.github.io/">
      <img src="https://img.shields.io/badge/Project-Page-blue?logo=googlechrome&logoColor=white" alt="Project Page" />
    </a>
    &nbsp;
    <a href="https://eccv.ecva.net/">
      <img src="https://img.shields.io/badge/ECCV-2026-blueviolet?logo=google-scholar&logoColor=white" alt="ECCV 2026" />
    </a>
    &nbsp;
    <a href="LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT" />
    </a>
  </p>

</div>

---

<p align="center">
  <b>ECCV 2026</b>
</p>

<p align="center">
  <a href="https://jwliao1209.github.io/">Jia-Wei Liao</a><sup>1,4</sup>,
  <a href="https://alexpeng517.github.io/">Li-Xuan Peng</a><sup>2</sup>,
  <a href="https://math.ntnu.edu.tw/~yueh/">Mei-Heng Yueh</a><sup>3</sup>,
  <a href="https://aliensunmin.github.io/">Min Sun</a><sup>2</sup>,
  <a href="https://www.csie.ntu.edu.tw/~ccf/">Cheng-Fu Chou</a><sup>1</sup>,
  <a href="https://homepage.citi.sinica.edu.tw/pages/pullpull/index_en.html">Jun-Cheng Chen</a><sup>4</sup>
  <br>
  <sup>1</sup>National Taiwan University,
  <sup>2</sup>National Tsing Hua University,
  <br>
  <sup>3</sup>National Taiwan Normal University,
  <sup>4</sup>Academia Sinica
</p>

### 💡 TL;DR

DiffRGD is a **plug-and-play, inference-time guidance framework** for pre-trained diffusion models. Existing guidance methods inject loss gradients directly into the sampling process, drifting the latent away from its per-step Gaussian distribution and degrading sample quality. DiffRGD instead formulates each DDIM step as a constrained optimization problem on a **spherical manifold induced by the latent Gaussian distribution** (via a polar decomposition of the isotropic Gaussian) and solves it with **Riemannian Gradient Descent** — projecting the guidance gradient onto the tangent space of the sphere and retracting the update back onto it. This preserves the latent distribution while steering the sample toward the condition, with a convergence guarantee to a stationary point.

---

## 🔧 Installation Guide

1. Clone this repository

```bash
git clone https://github.com/jwliao1209/DiffRGD
cd DiffRGD
```

2. Set up the environment with [uv](https://docs.astral.sh/uv/) (Python ≥ 3.13)

```bash
pip install uv
uv sync
```

All experiments run on a single GPU (we used an NVIDIA RTX 4090).

<details>
<summary>Show Repository Structure</summary>
<pre>
DiffRGD/
├── src/
│   ├── methods/               # Guidance methods: DiffRGD (ours), DPS, MPGD, DSG, ADMMDiff
│   ├── pipelines/
│   │   ├── pipeline_image_sampler.py      # Pixel-space DDIM pipeline with guidance
│   │   └── pipeline_sd_image_sampler.py   # Stable Diffusion pipeline with latent guidance
│   ├── losses/                # Guidance losses (inverse problem, face parsing/sketch/ID, style)
│   ├── operators.py           # Degradation operators A(.) for inverse problems
│   ├── metrics/               # PSNR, SSIM, LPIPS, FID, KID, mIoU, CLIP/style scores
│   ├── dataset.py             # Image and measurement datasets
│   └── model.py               # OpenAI guided-diffusion UNet (Diffusers-compatible)
├── configs/
│   ├── operators/             # Degradation operator configurations
│   └── prompts.json           # Text prompts for style-guided generation
├── evaluation/                # Metric computation for each experiment
├── scripts/                   # Downloads, checkpoint conversion, experiment runners, plots
├── style_images/              # WikiArt style references for style-guided generation
├── generate_dataset.py        # Build measurement datasets
├── run_inverse_problem.py     # Image restoration (Sec. 4.1, 4.2)
├── run_control.py             # Conditional generation (Sec. 4.3)
└── run_sd_style_transfer.py   # Style-guided generation with Stable Diffusion (Appendix E.2)
</pre>
</details>

## 🔗 Pre-trained Models

| Model | Used for | How to get it |
| --- | --- | --- |
| FFHQ 256 DDPM ([DPS](https://github.com/DPS2022/diffusion-posterior-sampling)) | FFHQ restoration | `bash scripts/download_model_checkpoint.sh` then convert (below) |
| ImageNet 256 DDPM ([guided-diffusion](https://github.com/openai/guided-diffusion)) | ImageNet restoration / denoising | same as above |
| CelebA-HQ 256 DDPM | Conditional generation | auto-downloaded (`google/ddpm-ema-celebahq-256`) |
| Stable Diffusion v1.5 | SD super-resolution / style transfer | auto-downloaded (`runwayml/stable-diffusion-v1-5`) |
| BiSeNet face parser | Segmentation guidance | place at `src/losses/checkpoints/bisnet.pth` (from [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)) |
| AODA sketch model | Sketch guidance | place at `src/losses/checkpoints/sketchnet.pth` (from [FreeDoM's model zoo](https://github.com/vvictoryuki/FreeDoM)) |
| ArcFace IR-SE50 | FaceID guidance | `bash scripts/download_face_id.sh` |

Convert the raw OpenAI checkpoints into Diffusers pipelines (creates `checkpoints/openai_ffhq` and `checkpoints/openai_imagenet`):

```bash
bash scripts/download_model_checkpoint.sh
uv run scripts/convert_openai_model_to_diffuser.py -d ffhq
uv run scripts/convert_openai_model_to_diffuser.py -d imagenet
```

## 📦 Datasets

Place validation images under `data/<dataset>/` (e.g. `data/ffhq/`, `data/imagenet/`, `data/celeba_hq/`). Download helpers:

```bash
uv run scripts/download_ffhq.py                  # FFHQ 256 (Kaggle)
uv run scripts/split_ffhq_valid.py               # split FFHQ into train/valid
uv run scripts/download_celeba_hq_dataset.py     # CelebA-HQ 256 (Kaggle)
bash scripts/download_imagenet_val_data.sh       # ImageNet validation set
```

## 🖼️ Image Restoration

Each task is a noisy linear inverse problem y = A(x) + n with n ~ N(0, 0.05²I). Tasks: `random_inpainting` (70% random mask), `box_inpainting`, `super_resolution` (4× bicubic), `gaussian_blur` (31×31, σ=3), `motion_blur` (61×61, intensity 0.5), and `denoising_<sigma>`.

**Generate measurements**, then **run** a method:

```bash
bash scripts/generate_dataset.sh 150 ffhq

uv run run_inverse_problem.py \
    -t random_inpainting \        # task
    -d ffhq \                     # ffhq | imagenet | celeba_hq
    -m diffrgd \                  # dps | mpgd | dsg | admmdiff | diffrgd
    -n 150 \                      # number of samples
    --num_inference_steps 100 \   # DDIM steps
    --step_size 5 \               # guidance strength eta_t
    -i 1 \                        # guidance interval
    --inner_max_iter 3 \          # RGD inner iterations K
    --noise gaussian
```

Reproduce the paper's tables with the tuned hyperparameters:

```bash
bash scripts/run_ffhq_t1000.sh 150      # Table 1: FFHQ, 1000 DDIM steps
bash scripts/run_ffhq_t100.sh 150       # Table 2: FFHQ, 100 DDIM steps
bash scripts/run_imagenet_t100.sh 150   # Table 3: ImageNet, 100 DDIM steps
bash scripts/run_denoising.sh 150       # Table 9: ImageNet denoising (σ = 0.1/0.2/0.3)
```

**Evaluate** (PSNR / SSIM / LPIPS / FID):

```bash
bash scripts/run_eval_ffhq.sh 150
bash scripts/run_eval_imagenet.sh 150
```

## 🔍 Super-Resolution with Stable Diffusion (8× / 12×)

FFHQ 1024×1024 images are downsampled to 64×64 and super-resolved to 512×512 (8×) or 768×768 (12×) with Stable Diffusion v1.5:

```bash
bash scripts/run_sd_super_resolution.sh 100
uv run evaluation/evaluate_sd_super_resolution.py -m diffrgd --resolution 512   # or 768
```

## 🎯 Conditional Generation

Face generation on CelebA-HQ guided by a **segmentation map** (BiSeNet), a **sketch** (AODA), or a **FaceID** embedding (ArcFace):

```bash
uv run run_control.py \
    -t segmentation \             # segmentation | sketch | face_id
    -m diffrgd \                  # dps (= FreeDoM) | mpgd | dsg | admmdiff | diffrgd
    -n 150 \
    --step_size 30 \
    --inner_max_iter 3
```

Reproduce Table 5 and evaluate (mIoU or condition-L2 / FID / KID):

```bash
bash scripts/run_control.sh 150
bash scripts/run_eval_control.sh 150
```

## 🎨 Style-Guided Generation

Style transfer with Stable Diffusion v1.5, using a CLIP Gram-matrix style loss over 5 WikiArt style images (`style_images/`) × 20 prompts (`configs/prompts.json`):

```bash
bash scripts/run_sd_style_transfer.sh
uv run evaluation/evaluate_sd_style_transfer.py -m diffrgd    # CLIP-score / Style-score (Table 10)
```

## 🙏 Acknowledgements

We are grateful for the foundational code provided by [DPS](https://github.com/DPS2022/diffusion-posterior-sampling), [FreeDoM](https://github.com/vvictoryuki/FreeDoM), [MPGD](https://github.com/KellyYutongHe/mpgd_pytorch), [DSG](https://github.com/LingxiaoYang2023/DSG2024), [ADMMDiff](https://github.com/youyuan-zhang/ADMMDiff), and 🤗 [Diffusers](https://github.com/huggingface/diffusers). Utilizing their resources implies agreement with their respective licenses. This research is supported by the National Science and Technology Council, Taiwan, and Academia Sinica.

## 📚 Citation

If you use our work or our implementation in this repo or find them helpful, please consider giving a citation.

```bibtex
@inproceedings{liao2026diffrgd,
  title     = {DiffRGD: An Inference-Time Diffusion Guidance Through Riemannian Gradient Descent},
  author    = {Liao, Jia-Wei and Peng, Li-Xuan and Yueh, Mei-Heng and Sun, Min and Chou, Cheng-Fu and Chen, Jun-Cheng},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```
