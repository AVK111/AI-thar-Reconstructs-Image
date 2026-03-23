"""
ensemble.py — Sequential ensemble inpainting pipeline

4 combinations:
    1. GAN + Autoencoder       (GAN → AE)
    2. GAN + Diffusion         (GAN → Diffusion)
    3. Autoencoder + Diffusion (AE  → Diffusion)
    4. All 3 combined          (AE  → GAN → Diffusion, PSNR-weighted blend)

Usage:
    python ensemble.py \
        --gan_model   models/gan/weights/gan/generator_best.keras \
        --ae_model    models/autoencoder/weights/autoencoder/autoencoder_best.keras \
        --diff_model  models/diffusion/weights/diffusion/unet_best.keras \
        --test_images test/ \
        --output_dir  ensemble_results/
"""

import sys
import time
import argparse
import csv
import numpy as np
import tensorflow as tf
from pathlib import Path
from scipy.ndimage import gaussian_filter

BASE = Path(__file__).resolve().parent
sys.path.append(str(BASE / "models" / "gan"))
sys.path.append(str(BASE / "models" / "diffusion"))

from dataset   import generate_freeform_mask
from scheduler import DDPMScheduler

IMAGE_SIZE = 256

# PSNR-based weights from individual model evaluation
# GAN=26.57, AE=23.49, Diff=14.63 → normalize
_TOTAL = 26.57 + 23.49 + 14.63
W_GAN  = 26.57 / _TOTAL   # 0.413
W_AE   = 23.49 / _TOTAL   # 0.365
W_DIFF = 14.63 / _TOTAL   # 0.227


# ──────────────────────────────────────────────
# Preprocessing / postprocessing
# ──────────────────────────────────────────────

def preprocess(image_np: np.ndarray, mask_np: np.ndarray):
    """uint8 (H,W,3) + uint8 (H,W) → float32 tensors at 256×256."""
    img = tf.image.resize(
        image_np[np.newaxis].astype(np.float32), [IMAGE_SIZE, IMAGE_SIZE]
    ) / 127.5 - 1.0

    if mask_np.ndim == 2:
        mask_np = mask_np[:, :, np.newaxis]
    msk = tf.cast(
        tf.image.resize(
            mask_np[np.newaxis].astype(np.float32), [IMAGE_SIZE, IMAGE_SIZE]
        ) / 255.0 > 0.5,
        tf.float32,
    )
    return img, msk


def postprocess(result_tf: tf.Tensor,
                orig_h: int, orig_w: int,
                mask_np: np.ndarray) -> np.ndarray:
    """[-1,1] tensor → uint8 (H,W,3) at original resolution + Gaussian smooth."""
    result = tf.image.resize(
        tf.cast(tf.clip_by_value((result_tf + 1.0) * 127.5, 0, 255), tf.uint8),
        [orig_h, orig_w],
    ).numpy().squeeze(0).astype(np.float32)

    smoothed = gaussian_filter(result, sigma=[0.6, 0.6, 0])

    if mask_np.ndim == 2:
        mask_np = mask_np[:, :, np.newaxis]
    msk = tf.image.resize(
        mask_np[np.newaxis].astype(np.float32), [orig_h, orig_w]
    ).numpy().squeeze(0)
    mbin = (msk / 255.0 > 0.5).astype(np.float32)

    blended = result * (1.0 - mbin) + smoothed * mbin
    return np.clip(blended, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────
# Single-model inference helpers
# ──────────────────────────────────────────────

def run_gan(model, img_tf, msk_tf) -> tf.Tensor:
    masked = img_tf * (1.0 - msk_tf)
    fake   = model(tf.concat([masked, msk_tf], axis=-1), training=False)
    return masked * (1.0 - msk_tf) + fake * msk_tf


def run_ae(model, img_tf, msk_tf) -> tf.Tensor:
    masked = img_tf * (1.0 - msk_tf)
    pred   = model(tf.concat([masked, msk_tf], axis=-1), training=False)
    return masked * (1.0 - msk_tf) + pred * msk_tf


def run_diffusion(model, scheduler, img_tf, msk_tf, steps=30) -> tf.Tensor:
    """
    Run RePaint diffusion with fewer steps (30) since we're only refining,
    not generating from scratch — the input is already partially inpainted.
    """
    x_t = tf.random.normal(tf.shape(img_tf))
    x_t = img_tf * (1.0 - msk_tf) + x_t * msk_tf

    step_size = scheduler.T // steps
    timesteps = list(range(scheduler.T - 1, -1, -step_size))

    for t in timesteps:
        t_batch    = tf.fill([tf.shape(img_tf)[0]], t)
        noise_pred = model([x_t, msk_tf, t_batch], training=False)
        x_t        = scheduler.step(x_t, t, noise_pred, img_tf, msk_tf)

    return tf.clip_by_value(x_t, -1.0, 1.0)


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_psnr(real, pred):
    r = (real + 1.0) / 2.0
    p = tf.clip_by_value((pred + 1.0) / 2.0, 0.0, 1.0)
    return float(tf.reduce_mean(tf.image.psnr(r, p, max_val=1.0)))

def compute_ssim(real, pred):
    r = (real + 1.0) / 2.0
    p = tf.clip_by_value((pred + 1.0) / 2.0, 0.0, 1.0)
    return float(tf.reduce_mean(tf.image.ssim(r, p, max_val=1.0)))

def compute_mae(real, pred, mask):
    diff = tf.abs(real - pred) * mask
    n    = tf.reduce_sum(mask) * 3.0
    return float(tf.reduce_sum(diff) / tf.maximum(n, 1.0))


# ──────────────────────────────────────────────
# 4 ensemble strategies
# ──────────────────────────────────────────────

def ensemble_gan_ae(gan, ae, img_tf, msk_tf) -> tf.Tensor:
    """GAN inpaints → AE refines the GAN output."""
    gan_out = run_gan(gan, img_tf, msk_tf)
    # Feed GAN output as "image" for AE — known pixels from GAN, AE refines hole
    ae_out  = run_ae(ae, gan_out, msk_tf)
    return ae_out


def ensemble_gan_diff(gan, diff_model, scheduler, img_tf, msk_tf) -> tf.Tensor:
    """GAN inpaints → Diffusion refines the GAN output."""
    gan_out  = run_gan(gan, img_tf, msk_tf)
    diff_out = run_diffusion(diff_model, scheduler, gan_out, msk_tf, steps=30)
    return diff_out


def ensemble_ae_diff(ae, diff_model, scheduler, img_tf, msk_tf) -> tf.Tensor:
    """AE inpaints (fast structural) → Diffusion refines."""
    ae_out   = run_ae(ae, img_tf, msk_tf)
    diff_out = run_diffusion(diff_model, scheduler, ae_out, msk_tf, steps=30)
    return diff_out


def ensemble_all3(gan, ae, diff_model, scheduler, img_tf, msk_tf) -> tf.Tensor:
    """
    All 3 sequential + PSNR-weighted blend:
        Step 1: AE  → rough structure
        Step 2: GAN → sharpen textures
        Step 3: Diffusion → final polish

    Final output = PSNR-weighted blend of all 3 results in the hole region.
    Known pixels from original image are always preserved.
    """
    ae_out   = run_ae(ae,    img_tf, msk_tf)
    gan_out  = run_gan(gan,  img_tf, msk_tf)
    diff_out = run_diffusion(diff_model, scheduler,
                             ae_out,    # start from AE output
                             msk_tf, steps=30)

    # PSNR-weighted blend inside the hole
    blended_hole = (W_GAN  * gan_out  +
                    W_AE   * ae_out   +
                    W_DIFF * diff_out)

    # Keep known pixels from original
    result = img_tf * (1.0 - msk_tf) + blended_hole * msk_tf
    return result


# ──────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────

def run_evaluation(args):
    print("\n" + "="*60)
    print("  ENSEMBLE INPAINTING EVALUATION")
    print("="*60)

    # ── Load models ──────────────────────────
    print("\n[1/3] Loading models …")
    gan       = tf.keras.models.load_model(args.gan_model,  compile=False)
    ae        = tf.keras.models.load_model(args.ae_model,   compile=False)
    diff      = tf.keras.models.load_model(args.diff_model, compile=False)
    scheduler = DDPMScheduler(T=1000)
    print("  All models loaded.")

    # ── Test images ──────────────────────────
    print("\n[2/3] Loading test images …")
    exts  = {".jpg", ".jpeg", ".png"}
    paths = [str(p) for p in Path(args.test_images).rglob("*")
             if p.suffix.lower() in exts][:args.max_images]
    if not paths:
        raise FileNotFoundError(f"No images in {args.test_images}")
    print(f"  {len(paths)} test images found.")

    # ── Output dirs ───────────────────────────
    out_dir = Path(args.output_dir)
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(exist_ok=True)

    # ── Strategies ───────────────────────────
    strategies = [
        "GAN + AE",
        "GAN + Diffusion",
        "AE + Diffusion",
        "GAN + AE + Diffusion",
    ]

    # ── Evaluation ───────────────────────────
    print("\n[3/3] Running ensemble inference …\n")
    print(f"{'Image':<20} {'Strategy':<22} {'PSNR':>7} {'SSIM':>6} {'Time':>7}")
    print("─" * 68)

    all_rows = []
    summary  = {s: [] for s in strategies}

    for idx, img_path in enumerate(paths):
        img_name = Path(img_path).stem

        # Load image
        raw    = tf.io.read_file(img_path)
        img_np = tf.image.decode_image(raw, channels=3,
                                       expand_animations=False).numpy()
        orig_h, orig_w = img_np.shape[:2]

        # Generate same mask for all strategies
        np.random.seed(idx)
        mask_np = (generate_freeform_mask() * 255).astype(np.uint8)

        img_tf, msk_tf = preprocess(img_np, mask_np)

        results = {}

        # 1. GAN + AE
        t0 = time.time()
        out = ensemble_gan_ae(gan, ae, img_tf, msk_tf)
        results["GAN + AE"] = (out, time.time() - t0)

        # 2. GAN + Diffusion
        t0 = time.time()
        out = ensemble_gan_diff(gan, diff, scheduler, img_tf, msk_tf)
        results["GAN + Diffusion"] = (out, time.time() - t0)

        # 3. AE + Diffusion
        t0 = time.time()
        out = ensemble_ae_diff(ae, diff, scheduler, img_tf, msk_tf)
        results["AE + Diffusion"] = (out, time.time() - t0)

        # 4. All 3
        t0 = time.time()
        out = ensemble_all3(gan, ae, diff, scheduler, img_tf, msk_tf)
        results["GAN + AE + Diffusion"] = (out, time.time() - t0)

        # Metrics + save
        for strategy, (out_tf, elapsed) in results.items():
            p = compute_psnr(img_tf, out_tf)
            s = compute_ssim(img_tf, out_tf)
            m = compute_mae(img_tf, out_tf, msk_tf)
            summary[strategy].append((p, s, m, elapsed))

            label = strategy.replace(" + ", "_").replace(" ", "_")
            out_np = postprocess(out_tf, orig_h, orig_w, mask_np)
            tf.io.write_file(
                str(img_dir / f"{img_name}_{label}.png"),
                tf.image.encode_png(tf.constant(out_np))
            )

            print(f"{img_name:<20} {strategy:<22} {p:>7.2f} {s:>6.4f} {elapsed:>6.2f}s")
            all_rows.append({
                "image": img_name, "strategy": strategy,
                "psnr": round(p, 4), "ssim": round(s, 4),
                "mae": round(m, 4), "time_sec": round(elapsed, 3)
            })
        print()

    # ── Summary ──────────────────────────────
    print("\n" + "="*60)
    print("  ENSEMBLE SUMMARY")
    print("="*60)
    print(f"\n{'Strategy':<25} {'PSNR':>8} {'SSIM':>8} {'MAE':>8} {'Time':>8}")
    print("─" * 62)

    # Include individual model baselines for comparison
    baselines = [
        ("GAN (individual)",         26.57, 0.9133, 0.1485, 0.69),
        ("Autoencoder (individual)", 23.49, 0.8874, 0.2488, 0.06),
        ("Diffusion (individual)",   14.63, 0.7684, 0.7318, 9.76),
    ]
    for name, p, s, m, t in baselines:
        print(f"{name:<25} {p:>8.2f} {s:>8.4f} {m:>8.4f} {t:>7.2f}s  [baseline]")
    print("─" * 62)

    summary_rows = []
    for strategy, metrics in summary.items():
        avg_p = np.mean([m[0] for m in metrics])
        avg_s = np.mean([m[1] for m in metrics])
        avg_m = np.mean([m[2] for m in metrics])
        avg_t = np.mean([m[3] for m in metrics])
        print(f"{strategy:<25} {avg_p:>8.2f} {avg_s:>8.4f} {avg_m:>8.4f} {avg_t:>7.2f}s")
        summary_rows.append({
            "strategy": strategy,
            "avg_psnr": round(avg_p, 4), "avg_ssim": round(avg_s, 4),
            "avg_mae":  round(avg_m, 4), "avg_time": round(avg_t, 3),
        })

    # ── Save CSVs ─────────────────────────────
    with open(out_dir / "ensemble_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader(); w.writerows(all_rows)

    with open(out_dir / "ensemble_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader(); w.writerows(summary_rows)

    print(f"\n  Results → {out_dir}/ensemble_results.csv")
    print(f"  Summary → {out_dir}/ensemble_summary.csv")
    print("="*60 + "\n")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate 4 ensemble combinations of GAN + AE + Diffusion"
    )
    parser.add_argument("--gan_model",   required=True)
    parser.add_argument("--ae_model",    required=True)
    parser.add_argument("--diff_model",  required=True)
    parser.add_argument("--test_images", required=True)
    parser.add_argument("--output_dir",  default="./ensemble_results")
    parser.add_argument("--max_images",  type=int, default=20)
    args = parser.parse_args()
    run_evaluation(args)
