"""
compare_models.py — Run all 3 inpainting models on the same test images
                    and generate a results table for the research paper.

Usage:
    python compare_models.py \
        --gan_model       models/gan/weights/gan/generator_best.keras \
        --ae_model        models/autoencoder/weights/autoencoder/autoencoder_best.keras \
        --diff_model      models/diffusion/weights/diffusion/unet_best.keras \
        --test_images     test_images/ \
        --output_dir      comparison_results/

Output:
    comparison_results/
        results_table.csv          ← import into paper (LaTeX/Word)
        results_summary.txt        ← human-readable summary
        images/
            <img>_masked.png
            <img>_gan.png
            <img>_autoencoder.png
            <img>_diffusion.png
            <img>_original.png
            <img>_grid.png         ← side-by-side comparison grid
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

# Add model dirs to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "models" / "gan"))
sys.path.append(str(BASE_DIR / "models" / "diffusion"))

from dataset   import generate_freeform_mask
from scheduler import DDPMScheduler


IMAGE_SIZE = 256


# ──────────────────────────────────────────────
# Image utilities
# ──────────────────────────────────────────────

def load_image(path: str) -> tf.Tensor:
    """Load image → (1, 256, 256, 3) float32 in [-1, 1]."""
    raw   = tf.io.read_file(path)
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return tf.expand_dims(image, 0)


def make_mask(seed: int = None) -> tf.Tensor:
    """Generate a reproducible free-form mask → (1, 256, 256, 1)."""
    if seed is not None:
        np.random.seed(seed)
    mask = generate_freeform_mask()
    return tf.expand_dims(tf.constant(mask), 0)


def to_uint8(tensor: tf.Tensor) -> np.ndarray:
    """Convert [-1,1] tensor → uint8 numpy (H, W, 3)."""
    t = tf.squeeze(tensor, 0)
    t = tf.cast(tf.clip_by_value((t + 1.0) * 127.5, 0, 255), tf.uint8)
    return t.numpy()


def save_png(array: np.ndarray, path: str):
    """Save uint8 (H, W, 3) numpy array as PNG."""
    t       = tf.constant(array)
    encoded = tf.image.encode_png(t)
    tf.io.write_file(path, encoded)


def make_grid(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
    """
    Create a horizontal comparison grid with labels.
    images : list of uint8 (H, W, 3) arrays
    """
    h, w   = images[0].shape[:2]
    n      = len(images)
    grid   = np.zeros((h, w * n, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        grid[:, i*w:(i+1)*w, :] = img
    return grid


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def psnr(real: tf.Tensor, pred: tf.Tensor) -> float:
    r = (real + 1.0) / 2.0
    p = tf.clip_by_value((pred + 1.0) / 2.0, 0.0, 1.0)
    return float(tf.reduce_mean(tf.image.psnr(r, p, max_val=1.0)))


def ssim(real: tf.Tensor, pred: tf.Tensor) -> float:
    r = (real + 1.0) / 2.0
    p = tf.clip_by_value((pred + 1.0) / 2.0, 0.0, 1.0)
    return float(tf.reduce_mean(tf.image.ssim(r, p, max_val=1.0)))


def mae(real: tf.Tensor, pred: tf.Tensor, mask: tf.Tensor) -> float:
    """Mean Absolute Error inside the masked (hole) region only."""
    diff    = tf.abs(real - pred) * mask
    n_pixels= tf.reduce_sum(mask) * 3.0
    return float(tf.reduce_sum(diff) / tf.maximum(n_pixels, 1.0))


# ──────────────────────────────────────────────
# Model inference functions
# ──────────────────────────────────────────────

def run_gan(model, image: tf.Tensor, mask: tf.Tensor) -> tuple[tf.Tensor, float]:
    """Run GAN generator and return composite + inference time."""
    t0        = time.time()
    gen_input = tf.concat([image * (1.0 - mask), mask], axis=-1)
    fake      = model(gen_input, training=False)
    composite = image * (1.0 - mask) + fake * mask
    elapsed   = time.time() - t0
    return composite, elapsed


def run_autoencoder(model, image: tf.Tensor, mask: tf.Tensor) -> tuple[tf.Tensor, float]:
    """Run Autoencoder and return composite + inference time."""
    t0        = time.time()
    ae_input  = tf.concat([image * (1.0 - mask), mask], axis=-1)
    predicted = model(ae_input, training=False)
    composite = image * (1.0 - mask) + predicted * mask
    elapsed   = time.time() - t0
    return composite, elapsed


def run_diffusion(
    model,
    scheduler: DDPMScheduler,
    image:     tf.Tensor,
    mask:      tf.Tensor,
    steps:     int = 50,
) -> tuple[tf.Tensor, float]:
    """Run Diffusion reverse process and return composite + inference time."""
    t0  = time.time()
    x_t = tf.random.normal(tf.shape(image))
    x_t = image * (1.0 - mask) + x_t * mask

    step_size = scheduler.T // steps
    timesteps = list(range(scheduler.T - 1, -1, -step_size))

    for t in timesteps:
        t_batch    = tf.fill([tf.shape(image)[0]], t)
        noise_pred = model([x_t, mask, t_batch], training=False)
        x_t        = scheduler.step(x_t, t, noise_pred, image, mask)

    composite = tf.clip_by_value(x_t, -1.0, 1.0)
    elapsed   = time.time() - t0
    return composite, elapsed


# ──────────────────────────────────────────────
# Main comparison runner
# ──────────────────────────────────────────────

def run_comparison(args):
    print("\n" + "="*60)
    print("  IMAGE INPAINTING — MODEL COMPARISON")
    print("="*60)

    # ── Load models ──────────────────────────
    print("\n[1/4] Loading models …")

    print(f"  Loading GAN          → {args.gan_model}")
    gan_model = tf.keras.models.load_model(args.gan_model, compile=False)

    print(f"  Loading Autoencoder  → {args.ae_model}")
    ae_model  = tf.keras.models.load_model(args.ae_model,  compile=False)

    print(f"  Loading Diffusion    → {args.diff_model}")
    diff_model = tf.keras.models.load_model(args.diff_model, compile=False)
    scheduler  = DDPMScheduler(T=1000)

    print("  All models loaded.")

    # ── Collect test images ───────────────────
    print("\n[2/4] Collecting test images …")
    exts       = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    test_paths = [
        str(p) for p in Path(args.test_images).rglob("*")
        if p.suffix in exts
    ][:args.max_images]

    if not test_paths:
        raise FileNotFoundError(f"No images found in {args.test_images}")

    print(f"  Found {len(test_paths)} test images")

    # ── Output directories ────────────────────
    out_dir = Path(args.output_dir)
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(exist_ok=True)

    # ── Per-image results ─────────────────────
    print("\n[3/4] Running inference on all models …\n")
    print(f"{'Image':<25} {'Model':<15} {'PSNR':>7} {'SSIM':>6} {'MAE':>7} {'Time':>7}")
    print("─" * 70)

    rows        = []
    all_metrics = {"GAN": [], "Autoencoder": [], "Diffusion": []}

    for idx, img_path in enumerate(test_paths):
        img_name = Path(img_path).stem
        image    = load_image(img_path)
        mask     = make_mask(seed=idx)           # same mask for all models

        masked_image = image * (1.0 - mask)
        results      = {}

        # ── GAN ──────────────────────────────
        gan_out, gan_time = run_gan(gan_model, image, mask)
        gan_psnr = psnr(image, gan_out)
        gan_ssim = ssim(image, gan_out)
        gan_mae  = mae(image,  gan_out, mask)
        results["GAN"] = (gan_out, gan_psnr, gan_ssim, gan_mae, gan_time)
        all_metrics["GAN"].append((gan_psnr, gan_ssim, gan_mae, gan_time))
        print(f"{img_name:<25} {'GAN':<15} {gan_psnr:>7.2f} {gan_ssim:>6.4f} {gan_mae:>7.4f} {gan_time:>6.2f}s")

        # ── Autoencoder ───────────────────────
        ae_out, ae_time = run_autoencoder(ae_model, image, mask)
        ae_psnr = psnr(image, ae_out)
        ae_ssim = ssim(image, ae_out)
        ae_mae  = mae(image,  ae_out, mask)
        results["Autoencoder"] = (ae_out, ae_psnr, ae_ssim, ae_mae, ae_time)
        all_metrics["Autoencoder"].append((ae_psnr, ae_ssim, ae_mae, ae_time))
        print(f"{'':25} {'Autoencoder':<15} {ae_psnr:>7.2f} {ae_ssim:>6.4f} {ae_mae:>7.4f} {ae_time:>6.2f}s")

        # ── Diffusion ─────────────────────────
        diff_out, diff_time = run_diffusion(diff_model, scheduler, image, mask, steps=args.diff_steps)
        diff_psnr = psnr(image, diff_out)
        diff_ssim = ssim(image, diff_out)
        diff_mae  = mae(image,  diff_out, mask)
        results["Diffusion"] = (diff_out, diff_psnr, diff_ssim, diff_mae, diff_time)
        all_metrics["Diffusion"].append((diff_psnr, diff_ssim, diff_mae, diff_time))
        print(f"{'':25} {'Diffusion':<15} {diff_psnr:>7.2f} {diff_ssim:>6.4f} {diff_mae:>7.4f} {diff_time:>6.2f}s")
        print()

        # ── Save per-image outputs ────────────
        orig_u8   = to_uint8(image)
        masked_u8 = to_uint8(masked_image)
        save_png(masked_u8,               str(img_dir / f"{img_name}_masked.png"))
        save_png(orig_u8,                 str(img_dir / f"{img_name}_original.png"))
        save_png(to_uint8(gan_out),       str(img_dir / f"{img_name}_gan.png"))
        save_png(to_uint8(ae_out),        str(img_dir / f"{img_name}_autoencoder.png"))
        save_png(to_uint8(diff_out),      str(img_dir / f"{img_name}_diffusion.png"))

        # ── Comparison grid ───────────────────
        grid = make_grid(
            [masked_u8,
             to_uint8(gan_out),
             to_uint8(ae_out),
             to_uint8(diff_out),
             orig_u8],
            ["Masked", "GAN", "Autoencoder", "Diffusion", "Original"]
        )
        save_png(grid, str(img_dir / f"{img_name}_grid.png"))

        # ── CSV rows ──────────────────────────
        for model_name, (_, mp, ms, mm, mt) in results.items():
            rows.append({
                "image":      img_name,
                "model":      model_name,
                "psnr_db":    round(mp, 4),
                "ssim":       round(ms, 4),
                "mae_hole":   round(mm, 4),
                "time_sec":   round(mt, 3),
            })

    # ── Aggregate results ─────────────────────
    print("\n[4/4] Computing aggregate results …\n")
    print("="*60)
    print(f"  SUMMARY  ({len(test_paths)} test images)")
    print("="*60)
    print(f"\n{'Model':<15} {'PSNR (dB)':>10} {'SSIM':>8} {'MAE':>8} {'Time/img':>10}")
    print("─" * 55)

    summary_rows = []
    for model_name, metrics in all_metrics.items():
        avg_psnr = np.mean([m[0] for m in metrics])
        avg_ssim = np.mean([m[1] for m in metrics])
        avg_mae  = np.mean([m[2] for m in metrics])
        avg_time = np.mean([m[3] for m in metrics])
        std_psnr = np.std([m[0]  for m in metrics])
        std_ssim = np.std([m[1]  for m in metrics])

        print(f"{model_name:<15} {avg_psnr:>8.2f}±{std_psnr:.2f} {avg_ssim:>6.4f}±{std_ssim:.4f} "
              f"{avg_mae:>8.4f} {avg_time:>8.2f}s")

        summary_rows.append({
            "model":       model_name,
            "avg_psnr":    round(avg_psnr, 4),
            "std_psnr":    round(std_psnr, 4),
            "avg_ssim":    round(avg_ssim, 4),
            "std_ssim":    round(std_ssim, 4),
            "avg_mae":     round(avg_mae,  4),
            "avg_time_sec":round(avg_time, 3),
            "n_images":    len(metrics),
        })

    # ── Save CSV ──────────────────────────────
    csv_path = out_dir / "results_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Per-image CSV  → {csv_path}")

    summary_path = out_dir / "results_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  Summary CSV    → {summary_path}")

    # ── Human-readable summary ────────────────
    txt_path = out_dir / "results_summary.txt"
    with open(txt_path, "w") as f:
        f.write("IMAGE INPAINTING — MODEL COMPARISON RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test images  : {len(test_paths)}\n")
        f.write(f"Image size   : {IMAGE_SIZE}×{IMAGE_SIZE}\n")
        f.write(f"Mask type    : Free-form (user-drawn simulation)\n")
        f.write(f"Diff steps   : {args.diff_steps} (DDPM inference)\n\n")
        f.write(f"{'Model':<15} {'PSNR (dB)':>12} {'SSIM':>10} {'MAE':>10} {'Time/img':>10}\n")
        f.write("-"*60 + "\n")
        for r in summary_rows:
            f.write(f"{r['model']:<15} "
                    f"{r['avg_psnr']:>7.2f}±{r['std_psnr']:.2f}  "
                    f"{r['avg_ssim']:>6.4f}±{r['std_ssim']:.4f}  "
                    f"{r['avg_mae']:>8.4f}  "
                    f"{r['avg_time_sec']:>7.2f}s\n")
        f.write("\nHigher PSNR and SSIM = better quality.\n")
        f.write("Lower MAE = better hole reconstruction.\n")
        f.write("Lower Time = faster inference.\n")

    print(f"  Text summary   → {txt_path}")
    print(f"  Image grids    → {img_dir}/")
    print("\n" + "="*60)
    print("  Comparison complete!")
    print("="*60 + "\n")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare GAN vs Autoencoder vs Diffusion for inpainting")

    parser.add_argument("--gan_model",    required=True, help="Path to generator_best.keras")
    parser.add_argument("--ae_model",     required=True, help="Path to autoencoder_best.keras")
    parser.add_argument("--diff_model",   required=True, help="Path to unet_best.keras")
    parser.add_argument("--test_images",  required=True, help="Folder of test images")
    parser.add_argument("--output_dir",   default="./comparison_results", help="Output folder")
    parser.add_argument("--max_images",   type=int, default=20,  help="Max test images to evaluate")
    parser.add_argument("--diff_steps",   type=int, default=50,  help="Diffusion inference steps")

    args = parser.parse_args()
    run_comparison(args)
