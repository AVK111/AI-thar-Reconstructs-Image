# 🖼️ AI-Powered Image Inpainting

> Reconstruct missing or damaged image regions using three generative deep learning models — GAN, Autoencoder, and Diffusion — with a full-stack interactive web application.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?style=flat-square&logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-15-black?style=flat-square&logo=next.js)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green?style=flat-square&logo=mongodb)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Model Results](#model-results)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Models](#models)
  - [GAN](#1-u-net-gan)
  - [Autoencoder](#2-convolutional-autoencoder)
  - [Diffusion](#3-denoising-diffusion-probabilistic-model-ddpm)
- [Ensemble Combinations](#ensemble-combinations)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training](#training)
- [Running the App](#running-the-app)
- [API Reference](#api-reference)
- [Tech Stack](#tech-stack)

---

## Overview

Image inpainting is the task of filling in missing or corrupted regions of an image in a visually coherent way. This project implements and compares **three generative deep learning architectures** for free-form inpainting:

| Approach | Strength |
|---|---|
| U-Net GAN | Best quality — adversarially trained with perceptual + style losses |
| Convolutional Autoencoder | Fastest — 163× lower latency than GAN, real-time capable |
| DDPM (Diffusion) | Most advanced architecture — iterative denoising with self-attention |

Users upload a damaged image, **paint over the region to restore** using an interactive canvas, select a model, and receive the inpainted result with quality metrics (PSNR, SSIM).

---

## Demo

```
User uploads image → draws mask on canvas → selects model → AI restores the region
```

**Web Application Features:**
- 🎨 HTML5 Canvas mask drawing with adjustable brush size and eraser
- 🤖 Choose between GAN, Autoencoder, or Diffusion model
- 📊 Real-time PSNR and SSIM metrics per result
- 🔄 Interactive before/after comparison slider
- 📁 Full job history with download capability

---

## Model Results

Evaluated on 20 held-out CelebA face images with identical free-form masks:

| Model | PSNR (dB) ↑ | SSIM ↑ | MAE ↓ | Time/img ↓ |
|---|---|---|---|---|
| **GAN** | **26.57** | **0.9133** | **0.1485** | 0.69s |
| Autoencoder | 23.49 | 0.8874 | 0.2488 | **0.06s** |
| Diffusion | 14.63 | 0.7684 | 0.7318 | 9.76s |

### Ensemble Results

| Strategy | PSNR (dB) | SSIM |
|---|---|---|
| GAN + AE | 23.49 | 0.8874 |
| GAN + Diffusion | 14.50 | 0.7678 |
| AE + Diffusion | 14.51 | 0.7677 |
| GAN + AE + Diffusion | 23.85 | 0.8207 |

> **Key finding:** Ensemble quality is bounded by the weakest component. GAN alone remains the best single model.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Next.js Frontend :3000                   │
│  Upload Image → Draw Mask → Select Model → View Result  │
└─────────────────────────┬───────────────────────────────┘
                          │ HTTP REST (Axios)
                          ▼
┌─────────────────────────────────────────────────────────┐
│                 FastAPI Backend :8000                    │
│                                                         │
│  POST /inpaint  GET /jobs  DELETE /jobs  GET /health    │
│                      │                                  │
│            InpaintingService                            │
│                      │                                  │
│         InpaintingPipeline (Singleton)                  │
│        ┌─────────┬──────────────┬──────────┐           │
│        │   GAN   │ Autoencoder  │ Diffusion │           │
│        └─────────┴──────────────┴──────────┘           │
│                                                         │
│    MongoDB Atlas          /storage (files)              │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
   CelebA (202,599)             ImageNet (539,826)
   face images                  diverse scenes
```

---

## Project Structure

```
DL_PBL/
│
├── my-app-backend/                 # FastAPI backend
│   ├── app/
│   │   ├── main.py                 # FastAPI entrypoint
│   │   ├── core/
│   │   │   ├── config.py           # Settings and env vars
│   │   │   └── pipeline.py         # Model singleton + inference
│   │   ├── api/v1/
│   │   │   ├── inpaint.py          # Inpainting routes
│   │   │   └── jobs.py             # Job history routes
│   │   ├── db/
│   │   │   ├── client.py           # MongoDB connection
│   │   │   └── repositories.py     # CRUD operations
│   │   ├── schemas/
│   │   │   └── inpaint.py          # Pydantic models
│   │   ├── services/
│   │   │   └── inpaint_service.py  # Business logic
│   │   └── utils/
│   │       └── image_utils.py      # Image helpers
│   │
│   ├── models/
│   │   ├── gan/
│   │   │   ├── generator.py        # U-Net generator
│   │   │   ├── discriminator.py    # PatchGAN discriminators
│   │   │   ├── losses.py           # Combined loss functions
│   │   │   ├── dataset.py          # Data pipeline
│   │   │   ├── train.py            # Training script
│   │   │   └── infer.py            # Inference script
│   │   │
│   │   ├── autoencoder/
│   │   │   ├── encoder.py
│   │   │   ├── decoder.py
│   │   │   ├── train.py
│   │   │   └── infer.py
│   │   │
│   │   └── diffusion/
│   │       ├── unet.py             # Conditional U-Net denoiser
│   │       ├── scheduler.py        # DDPM scheduler
│   │       ├── train.py
│   │       └── infer.py
│   │
│   ├── ensemble.py                 # 4 ensemble combinations
│   ├── compare_models.py           # Model comparison script
│   ├── storage/
│   │   ├── uploads/
│   │   ├── masks/
│   │   └── outputs/
│   └── .env                        # Environment variables
│
└── my-app-frontend/                # Next.js frontend
    ├── app/
    │   ├── page.tsx                # Landing page
    │   ├── layout.tsx              # Root layout + navbar
    │   ├── globals.css             # Global styles
    │   ├── inpaint/
    │   │   └── page.tsx            # Main inpainting tool
    │   └── history/
    │       └── page.tsx            # Job history page
    ├── components/
    │   ├── canvas/
    │   │   └── MaskCanvas.tsx      # HTML5 Canvas mask drawing
    │   ├── results/
    │   │   └── CompareSlider.tsx   # Before/after slider
    │   └── ui/
    │       └── Navbar.tsx          # Navigation bar
    └── lib/
        └── api.ts                  # API client (Axios)
```

---

## Models

### 1. U-Net GAN

**Architecture:**
- **Generator:** U-Net encoder-decoder with 6 encoder blocks (Conv2D, BatchNorm, LeakyReLU) downsampling 256×256 → 4×4, a dilated gated convolution bottleneck (7 blocks, dilation rates [1,2,4,8,4,2,1]), and 6 decoder blocks with skip connections
- **Discriminators:** Dual PatchGAN — global (256×256 → 16×16 patch map) + local (128×128 mask crop → 8×8 patch map)
- **Gated convolutions:** `output = LeakyReLU(f(x)) ⊙ σ(g(x))` — learns to ignore masked pixels automatically

**Loss Function:**
```
L_G = L_adv + 30·L1_hole + 10·L1_valid + 0.1·L_perceptual + 50·L_style
```
- VGG16 perceptual and Gram matrix style losses at `relu1_2`, `relu2_2`, `relu3_3`

**Training:** 100 epochs · batch size 4 · Adam lr=2e-4 · β₁=0.5 · ~105s/epoch

---

### 2. Convolutional Autoencoder

**Architecture:**
- **Encoder:** 6 strided Conv2D blocks → 4×4×512 → Dense → 512-dim latent vector
- **Decoder:** Dense → reshape → 6 ConvTranspose2D blocks → 256×256×3 → tanh

**Loss Function:**
```
L_AE = 6·L1_hole + L1_valid + 0.05·L_perceptual
```

**Training:** 88 epochs · batch size 4 · cosine LR decay from 1e-3 · ~70s/epoch

---

### 3. Denoising Diffusion Probabilistic Model (DDPM)

**Architecture:**
- **Forward process:** Linear beta schedule (β₁=1e-4, β_T=0.02, T=1000), noise added only to masked region
- **Denoiser:** Conditional U-Net with sinusoidal timestep embedding injected at every residual block, self-attention at 16×16 spatial resolution
- **Inference:** RePaint strategy — 50 subsampled denoising steps with known pixel resampling at each step

**Training:** 50 epochs · batch size 2 · Adam lr=1e-4 · ~120s/epoch

---

## Ensemble Combinations

Four sequential ensemble pipelines were evaluated:

```python
# Strategy 1: GAN → AE
gan_out = gan(masked_image)
final   = ae(gan_out)          # AE refines GAN output

# Strategy 2: GAN → Diffusion  
gan_out  = gan(masked_image)
final    = diffusion(gan_out, steps=30)   # Diffusion polishes GAN

# Strategy 3: AE → Diffusion
ae_out = ae(masked_image)
final  = diffusion(ae_out, steps=30)

# Strategy 4: AE → GAN → Diffusion (PSNR-weighted blend)
ae_out   = ae(masked_image)
gan_out  = gan(masked_image)
diff_out = diffusion(ae_out, steps=30)
final    = 0.413*gan_out + 0.365*ae_out + 0.227*diff_out  # PSNR weights
```

---

## Dataset

| Dataset | Images | Content |
|---|---|---|
| CelebA | 202,599 | Aligned celebrity face images |
| ImageNet | 539,826 | 1,000-class diverse scenes |
| **Total** | **742,425** | Combined training set |

**Preprocessing:**
- Resize to 256×256
- Normalize to [-1, 1] via `x = (pixel / 127.5) - 1.0`
- Free-form masks: 5 random brush strokes per image (length 10–80px, width 8–25px)

---

## Installation

### Prerequisites
- Python 3.10
- Node.js 20+
- NVIDIA GPU (recommended) with CUDA
- MongoDB Atlas account

### Backend Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/ai-image-inpainting.git
cd ai-image-inpainting/my-app-backend

# Create conda environment
conda create -n inpainting python=3.10
conda activate inpainting

# Install dependencies
pip install tensorflow==2.13
pip install fastapi uvicorn motor pymongo python-dotenv
pip install pillow scipy python-multipart axios
```

Create `.env` file in `my-app-backend/`:
```env
MONGODB_URL=mongodb+srv://<user>:<password>@cluster0.xxxxx.mongodb.net/?appName=Cluster0
DB_NAME=inpainting_db
GAN_MODEL_PATH=models/gan/weights/gan/generator_best.keras
AE_MODEL_PATH=models/autoencoder/weights/autoencoder/autoencoder_best.keras
DIFF_MODEL_PATH=models/diffusion/weights/diffusion/unet_best.keras
STORAGE_PATH=storage
```

### Frontend Setup

```bash
cd ../my-app-frontend

# Install dependencies
npm install
npm install axios lucide-react react-before-after-slider-component --legacy-peer-deps
```

Create `.env.local` in `my-app-frontend/`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Training

### Train GAN
```bash
cd my-app-backend
python models/gan/train.py \
  --dataset combined \
  --epochs 100 \
  --batch_size 4 \
  --max_images 1000 \
  --save_every 5 \
  --checkpoint_dir ./models/gan/weights/gan
```

### Train Autoencoder
```bash
python models/autoencoder/train.py \
  --dataset combined \
  --epochs 88 \
  --batch_size 4 \
  --max_images 1000 \
  --save_every 5 \
  --checkpoint_dir ./models/autoencoder/weights/autoencoder
```

### Train Diffusion
```bash
python models/diffusion/train.py \
  --dataset combined \
  --epochs 50 \
  --batch_size 2 \
  --max_images 500 \
  --save_every 5 \
  --checkpoint_dir ./models/diffusion/weights/diffusion
```

### Run Model Comparison
```bash
python compare_models.py \
  --gan_model models/gan/weights/gan/generator_best.keras \
  --ae_model models/autoencoder/weights/autoencoder/autoencoder_best.keras \
  --diff_model models/diffusion/weights/diffusion/unet_best.keras \
  --test_images test/test_images/ \
  --output_dir comparison_results/ \
  --max_images 20
```

### Run Ensemble Evaluation
```bash
python ensemble.py \
  --gan_model models/gan/weights/gan/generator_best.keras \
  --ae_model models/autoencoder/weights/autoencoder/autoencoder_best.keras \
  --diff_model models/diffusion/weights/diffusion/unet_best.keras \
  --test_images test/test_images/ \
  --output_dir ensemble_results/ \
  --max_images 20
```

---

## Running the App

### Start Backend
```bash
cd my-app-backend
conda activate inpainting
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

All 3 models load into GPU memory at startup. You should see:
```
[pipeline] GAN loaded ✓
[pipeline] Autoencoder loaded ✓
[pipeline] Diffusion loaded ✓
[startup] API ready.
```

### Start Frontend
```bash
cd my-app-frontend
npm run dev
```

Open `http://localhost:3000` in your browser.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/inpaint` | Submit inpainting job (image + mask + model) |
| `GET` | `/api/v1/inpaint/{job_id}` | Get job result by ID |
| `GET` | `/api/v1/result/{filename}` | Download output image |
| `GET` | `/api/v1/jobs` | List all past jobs |
| `DELETE` | `/api/v1/jobs/{job_id}` | Delete job and files |
| `GET` | `/api/v1/health` | Health check + model status |

**Example request:**
```bash
curl -X POST http://localhost:8000/api/v1/inpaint \
  -F "image=@photo.jpg" \
  -F "mask=@mask.png" \
  -F "model=gan" \
  -F "diff_steps=50"
```

**Example response:**
```json
{
  "id": "64f3a...",
  "status": "completed",
  "model": "gan",
  "psnr": 26.57,
  "ssim": 0.9133,
  "time_sec": 0.69,
  "output_file": "output_gan_abc123.png"
}
```

---

## Tech Stack

**Backend**
- Python 3.10 · TensorFlow 2.13 · Keras
- FastAPI · Uvicorn
- Motor (async MongoDB) · MongoDB Atlas
- SciPy · Pillow · NumPy

**Frontend**
- Next.js 15 · TypeScript · Tailwind CSS
- Axios · Lucide React

**ML Models**
- U-Net GAN with Gated Convolutions
- Convolutional Autoencoder
- DDPM with RePaint Inference
- VGG16 (perceptual loss)

**Hardware**
- NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
- Training: ~3–4 hours per model

---

## Authors

| Name | Email |
|---|---|
| Atharv Jadhav | atharv.22310617@viit.ac.in |
| Aditya Sirsat | aditya.22310911@viit.ac.in |
| Atharv Kale | atharv.22311072@viit.ac.in |


---

## Acknowledgements

- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) — S. Liu et al., ICCV 2015
- [ImageNet](https://www.image-net.org/) — J. Deng et al., CVPR 2009
- [RePaint](https://arxiv.org/abs/2201.09865) — Lugmayr et al., CVPR 2022
- [DeepFill v2](https://arxiv.org/abs/1806.03589) — Yu et al., ICCV 2019

---

*Research paper published at International Conference on Deep Learning and Computer Vision, 2026*
