"""
train.py — DDPM training for image inpainting

Usage:
    # Smoke test
    python train.py --epochs 2 --batch_size 2 --max_images 50 --save_every 1

    # Full training (recommended — faster than GAN)
    python train.py --epochs 50 --batch_size 4 --max_images 500 --save_every 5
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).parents[1] / "gan"))
from dataset import make_combined_dataset, make_celeba_dataset, make_imagenet_dataset

from unet      import build_unet
from scheduler import DDPMScheduler


# ──────────────────────────────────────────────
# Device setup
# ──────────────────────────────────────────────

def configure_device():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[train] {len(gpus)} GPU(s) detected.")
    else:
        print("[train] Running on CPU.")
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(4)


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_psnr(real: tf.Tensor, fake: tf.Tensor) -> float:
    r = (real + 1.0) / 2.0
    f = tf.clip_by_value((fake + 1.0) / 2.0, 0.0, 1.0)
    return float(tf.reduce_mean(tf.image.psnr(r, f, max_val=1.0)))

def compute_ssim(real: tf.Tensor, fake: tf.Tensor) -> float:
    r = (real + 1.0) / 2.0
    f = tf.clip_by_value((fake + 1.0) / 2.0, 0.0, 1.0)
    return float(tf.reduce_mean(tf.image.ssim(r, f, max_val=1.0)))


# ──────────────────────────────────────────────
# Training step
# ──────────────────────────────────────────────

@tf.function
def train_step(real_image, mask, unet, scheduler_consts, optimizer):
    """
    One DDPM training step:
    1. Sample random timestep t
    2. Add noise only to the masked region → x_t
    3. Predict the noise with U-Net
    4. Compute MSE loss between predicted and actual noise
    """
    sqrt_ab, sqrt_1mab, betas, T = scheduler_consts
    batch_size = tf.shape(real_image)[0]

    # Sample random timesteps
    t = tf.random.uniform((batch_size,), 0, T, dtype=tf.int32)

    # Sample noise
    noise = tf.random.normal(tf.shape(real_image))

    # Add noise only to masked region
    sqrt_ab_t   = tf.gather(sqrt_ab,   t)[:, None, None, None]
    sqrt_1mab_t = tf.gather(sqrt_1mab, t)[:, None, None, None]
    x_t_full    = sqrt_ab_t * real_image + sqrt_1mab_t * noise     # full noisy image
    x_t         = real_image * (1.0 - mask) + x_t_full * mask      # noise only in hole

    with tf.GradientTape() as tape:
        noise_pred = unet([x_t, mask, t], training=True)

        # Only supervise noise prediction inside the masked region
        noise_in_hole = noise * mask
        pred_in_hole  = noise_pred * mask
        loss          = tf.reduce_mean(tf.square(noise_in_hole - pred_in_hole))

    grads = tape.gradient(loss, unet.trainable_variables)
    optimizer.apply_gradients(zip(grads, unet.trainable_variables))
    return loss, x_t


# ──────────────────────────────────────────────
# Inference — full reverse diffusion
# ──────────────────────────────────────────────

def ddpm_inpaint(
    unet:      tf.keras.Model,
    scheduler: DDPMScheduler,
    x0_known:  tf.Tensor,
    mask:      tf.Tensor,
    T_inf:     int = 200,
) -> tf.Tensor:
    """
    Run reverse diffusion to inpaint the masked region.

    Uses a reduced number of steps (T_inf=200 instead of 1000) for
    faster inference while maintaining quality.

    Args:
        x0_known : clean original image  (B, H, W, 3)
        mask     : binary mask           (B, H, W, 1)  1=hole
        T_inf    : inference steps (fewer = faster, more = better quality)

    Returns:
        inpainted : (B, H, W, 3) in [-1, 1]
    """
    batch_size = tf.shape(x0_known)[0]

    # Start from pure noise in the hole region
    x_t = tf.random.normal(tf.shape(x0_known))
    x_t = x0_known * (1.0 - mask) + x_t * mask

    # Subsample timesteps for faster inference
    step_size = scheduler.T // T_inf
    timesteps = list(range(scheduler.T - 1, -1, -step_size))

    for i, t in enumerate(timesteps):
        t_batch     = tf.fill([batch_size], t)
        noise_pred  = unet([x_t, mask, t_batch], training=False)
        x_t         = scheduler.step(x_t, t, noise_pred, x0_known, mask)

        if i % 40 == 0:
            print(f"  [infer] denoising step {i+1}/{len(timesteps)}", end="\r")

    print()
    return tf.clip_by_value(x_t, -1.0, 1.0)


# ──────────────────────────────────────────────
# Sample saver
# ──────────────────────────────────────────────

def save_samples(unet, scheduler, sample_batch, epoch, out_dir, T_inf=50):
    masked_image, mask, real_image = sample_batch
    n = min(2, masked_image.shape[0])   # only 2 samples for speed

    print(f"  [sample] running inference for epoch {epoch} sample …")
    composite = ddpm_inpaint(unet, scheduler, real_image[:n], mask[:n], T_inf=T_inf)

    def to_uint8(t):
        return tf.cast(tf.clip_by_value((t + 1.0) * 127.5, 0, 255), tf.uint8)

    row  = tf.concat([to_uint8(masked_image[:n]), to_uint8(composite), to_uint8(real_image[:n])], axis=2)
    grid = tf.concat(tf.unstack(row, axis=0), axis=0)
    path = os.path.join(out_dir, f"epoch_{epoch:04d}.png")
    tf.io.write_file(path, tf.image.encode_png(grid))
    print(f"  [sample] saved → {path}")
    return composite, real_image[:n]


# ──────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────

def train(args):
    configure_device()

    # ── Dataset ──────────────────────────────
    if args.dataset == "combined":
        dataset = make_combined_dataset(batch_size=args.batch_size, celeba_csv=args.celeba_csv, imagenet_dir=args.imagenet_dir)
    elif args.dataset == "celeba":
        dataset = make_celeba_dataset(batch_size=args.batch_size, csv_path=args.celeba_csv)
    else:
        dataset = make_imagenet_dataset(batch_size=args.batch_size, imagenet_dir=args.imagenet_dir)

    if args.max_images > 0:
        steps_limit = max(1, args.max_images // args.batch_size)
        dataset     = dataset.take(steps_limit)
        print(f"[train] Using {steps_limit} steps ({args.max_images} images) per epoch")

    sample_batch = next(iter(dataset))

    # ── Scheduler ────────────────────────────
    scheduler = DDPMScheduler(T=args.T)

    # Pack scheduler constants for use inside @tf.function
    scheduler_consts = (
        tf.constant(scheduler.sqrt_alpha_bars),
        tf.constant(scheduler.sqrt_one_minus_ab),
        tf.constant(scheduler.betas),
        args.T,
    )

    # ── Model ────────────────────────────────
    unet = build_unet(base_channels=args.base_channels)

    # ── Optimizer ────────────────────────────
    total_steps = args.epochs * max(1, args.max_images // args.batch_size)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=total_steps,
        alpha=1e-6,
    )
    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    # ── Checkpoint ───────────────────────────
    ckpt_dir   = Path(args.checkpoint_dir)
    sample_dir = ckpt_dir / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(exist_ok=True)

    checkpoint   = tf.train.Checkpoint(unet=unet, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, str(ckpt_dir), max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print(f"[train] Restored: {ckpt_manager.latest_checkpoint}")

    writer = tf.summary.create_file_writer(str(ckpt_dir / "logs"))

    # ── Training ─────────────────────────────
    print(f"\n[train] Diffusion DDPM — {args.epochs} epochs  T={args.T}\n")
    print(f"{'Epoch':>6} | {'MSE Loss':>10} | {'Time':>7}")
    print("─" * 35)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        total_loss = 0.0
        steps      = 0

        for masked_image, mask, real_image in dataset:
            loss, _ = train_step(real_image, mask, unet, scheduler_consts, optimizer)
            total_loss += float(loss)
            steps      += 1

            if steps % 10 == 0:
                print(f"  step {steps:4d} | loss={total_loss/steps:.5f}", end="\r")

        avg_loss = total_loss / max(steps, 1)
        elapsed  = time.time() - t0

        print(f"\r{epoch:6d} | {avg_loss:10.6f} | {elapsed:6.1f}s")

        with writer.as_default():
            tf.summary.scalar("loss/mse", avg_loss, step=epoch)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_samples(unet, scheduler, sample_batch, epoch, str(sample_dir), T_inf=args.T_inf)
            ckpt_manager.save()
            print(f"  [ckpt] saved at epoch {epoch}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            unet.save(str(ckpt_dir / "unet_best.keras"))
            print(f"  [best] loss={best_loss:.6f} → unet_best.keras")

    print("\n[train] Done.")
    unet.save(str(ckpt_dir / "unet_final.keras"))
    print(f"[train] Final model → {ckpt_dir / 'unet_final.keras'}")
    print(f"[train] Best MSE loss: {best_loss:.6f}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diffusion model for image inpainting")
    parser.add_argument("--dataset",        type=str,   default="combined", choices=["combined","celeba","imagenet"])
    parser.add_argument("--celeba_csv",     type=str,   default=None)
    parser.add_argument("--imagenet_dir",   type=str,   default=None)
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--batch_size",     type=int,   default=2)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--T",              type=int,   default=1000,  help="Total diffusion timesteps")
    parser.add_argument("--T_inf",          type=int,   default=50,    help="Inference steps (fewer=faster)")
    parser.add_argument("--base_channels",  type=int,   default=32,    help="U-Net base channels (32 for CPU, 64 for GPU)")
    parser.add_argument("--max_images",     type=int,   default=500)
    parser.add_argument("--save_every",     type=int,   default=5)
    parser.add_argument("--checkpoint_dir", type=str,   default="./weights/diffusion")
    args = parser.parse_args()
    train(args)
