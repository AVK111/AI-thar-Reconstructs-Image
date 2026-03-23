"""
train.py — Autoencoder training for image inpainting

Much simpler than GAN — single model, single loss, faster training.

Usage:
    # Smoke test
    python train.py --epochs 2 --batch_size 2 --max_images 50 --save_every 1

    # Full training (recommended)
    python train.py --epochs 50 --batch_size 4 --max_images 500 --save_every 5
"""

import argparse
import os
import sys
import time
from pathlib import Path

import tensorflow as tf

sys.path.append(str(Path(__file__).parents[1] / "gan"))
from dataset import make_combined_dataset, make_celeba_dataset, make_imagenet_dataset

from decoder import build_autoencoder, inpaint_composite


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
# Losses
# ──────────────────────────────────────────────

def ae_loss(
    real:      tf.Tensor,
    predicted: tf.Tensor,
    mask:      tf.Tensor,
) -> dict:
    """
    Combined loss for autoencoder inpainting:

    1. Hole L1     : heavy penalty on the masked region (weight 6.0)
    2. Valid L1    : light penalty on known pixels (weight 1.0)
                     forces the model to reconstruct full image coherently
    3. Perceptual  : VGG feature-level similarity (weight 0.05)
                     encourages realistic textures without adversarial training

    Returns dict with total + per-component losses.
    """
    diff       = tf.abs(real - predicted)
    hole_loss  = tf.reduce_mean(diff * mask)         * 6.0
    valid_loss = tf.reduce_mean(diff * (1.0 - mask)) * 1.0

    # Simple perceptual loss via VGG16
    perc_loss  = perceptual_loss(real, predicted)    * 0.05

    total = hole_loss + valid_loss + perc_loss
    return {
        "total":   total,
        "hole":    hole_loss,
        "valid":   valid_loss,
        "perc":    perc_loss,
    }


# ── VGG perceptual loss ───────────────────────

_vgg = None

def _get_vgg():
    global _vgg
    if _vgg is None:
        vgg  = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
        vgg.trainable = False
        outs = [vgg.get_layer(n).output for n in ["block1_conv2", "block2_conv2", "block3_conv3"]]
        _vgg = tf.keras.Model(vgg.input, outs)
    return _vgg

def perceptual_loss(real: tf.Tensor, fake: tf.Tensor) -> tf.Tensor:
    def preprocess(x):
        x = (x + 1.0) * 127.5
        return tf.keras.applications.vgg16.preprocess_input(x)
    vgg    = _get_vgg()
    r_feat = vgg(preprocess(real), training=False)
    f_feat = vgg(preprocess(fake), training=False)
    return sum(tf.reduce_mean(tf.abs(r - f)) for r, f in zip(r_feat, f_feat))


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_psnr(real: tf.Tensor, fake: tf.Tensor) -> float:
    r = (real + 1.0) / 2.0
    f = (fake + 1.0) / 2.0
    return float(tf.reduce_mean(tf.image.psnr(r, f, max_val=1.0)))

def compute_ssim(real: tf.Tensor, fake: tf.Tensor) -> float:
    r = (real + 1.0) / 2.0
    f = (fake + 1.0) / 2.0
    return float(tf.reduce_mean(tf.image.ssim(r, f, max_val=1.0)))


# ──────────────────────────────────────────────
# Training step
# ──────────────────────────────────────────────

@tf.function
def train_step(masked_image, mask, real_image, autoencoder, optimizer):
    with tf.GradientTape() as tape:
        ae_input  = tf.concat([masked_image, mask], axis=-1)
        predicted = autoencoder(ae_input, training=True)
        composite = masked_image * (1.0 - mask) + predicted * mask
        losses    = ae_loss(real_image, composite, mask)

    grads = tape.gradient(losses["total"], autoencoder.trainable_variables)
    optimizer.apply_gradients(zip(grads, autoencoder.trainable_variables))
    return losses, composite


# ──────────────────────────────────────────────
# Sample saver
# ──────────────────────────────────────────────

def save_samples(autoencoder, sample_batch, epoch, out_dir):
    masked_image, mask, real_image = sample_batch
    n         = min(4, masked_image.shape[0])
    ae_input  = tf.concat([masked_image[:n], mask[:n]], axis=-1)
    predicted = autoencoder(ae_input, training=False)
    composite = masked_image[:n] * (1.0 - mask[:n]) + predicted * mask[:n]

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

    # ── Model ────────────────────────────────
    autoencoder = build_autoencoder(latent_dim=args.latent_dim)

    # ── Optimizer — Adam with cosine decay ───
    # Autoencoder converges faster than GAN — use cosine LR schedule
    total_steps = args.epochs * max(1, args.max_images // args.batch_size)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=total_steps,
        alpha=1e-6,
    )
    optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9)

    # ── Checkpoint ───────────────────────────
    ckpt_dir   = Path(args.checkpoint_dir)
    sample_dir = ckpt_dir / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(exist_ok=True)

    checkpoint   = tf.train.Checkpoint(autoencoder=autoencoder, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, str(ckpt_dir), max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print(f"[train] Restored: {ckpt_manager.latest_checkpoint}")

    # ── TensorBoard ──────────────────────────
    writer = tf.summary.create_file_writer(str(ckpt_dir / "logs"))

    # ── Training ─────────────────────────────
    print(f"\n[train] Autoencoder — {args.epochs} epochs\n")
    print(f"{'Epoch':>6} | {'Loss':>8} {'Hole':>8} {'Valid':>8} {'Perc':>8} | {'PSNR':>7} {'SSIM':>6} | {'Time':>7}")
    print("─" * 75)

    best_psnr = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        totals    = {"total": 0.0, "hole": 0.0, "valid": 0.0, "perc": 0.0}
        psnr_list, ssim_list = [], []
        steps = 0

        for masked_image, mask, real_image in dataset:
            losses, composite = train_step(masked_image, mask, real_image, autoencoder, optimizer)
            for k in totals:
                totals[k] += float(losses[k])
            psnr_list.append(compute_psnr(real_image, composite))
            ssim_list.append(compute_ssim(real_image, composite))
            steps += 1

            if steps % 10 == 0:
                print(f"  step {steps:4d} | loss={totals['total']/steps:.3f} | "
                      f"PSNR={sum(psnr_list)/len(psnr_list):.2f}dB | "
                      f"SSIM={sum(ssim_list)/len(ssim_list):.4f}", end="\r")

        avg      = {k: totals[k] / max(steps, 1) for k in totals}
        avg_psnr = sum(psnr_list) / max(len(psnr_list), 1)
        avg_ssim = sum(ssim_list) / max(len(ssim_list), 1)
        elapsed  = time.time() - t0

        print(f"\r{epoch:6d} | {avg['total']:8.4f} {avg['hole']:8.4f} {avg['valid']:8.4f} {avg['perc']:8.4f} | "
              f"{avg_psnr:7.2f} {avg_ssim:6.4f} | {elapsed:6.1f}s")

        with writer.as_default():
            for k, v in avg.items():
                tf.summary.scalar(f"loss/{k}", v, step=epoch)
            tf.summary.scalar("metrics/psnr", avg_psnr, step=epoch)
            tf.summary.scalar("metrics/ssim", avg_ssim, step=epoch)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_samples(autoencoder, sample_batch, epoch, str(sample_dir))
            ckpt_manager.save()
            print(f"  [ckpt] saved at epoch {epoch}")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            autoencoder.save(str(ckpt_dir / "autoencoder_best.keras"))
            print(f"  [best] PSNR={best_psnr:.2f}dB → autoencoder_best.keras")

    print("\n[train] Done.")
    autoencoder.save(str(ckpt_dir / "autoencoder_final.keras"))
    print(f"[train] Final model → {ckpt_dir / 'autoencoder_final.keras'}")
    print(f"[train] Best PSNR: {best_psnr:.2f} dB")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Autoencoder for image inpainting")
    parser.add_argument("--dataset",        type=str,   default="combined", choices=["combined","celeba","imagenet"])
    parser.add_argument("--celeba_csv",     type=str,   default=None)
    parser.add_argument("--imagenet_dir",   type=str,   default=None)
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--batch_size",     type=int,   default=4)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--latent_dim",     type=int,   default=512)
    parser.add_argument("--max_images",     type=int,   default=500)
    parser.add_argument("--save_every",     type=int,   default=5)
    parser.add_argument("--checkpoint_dir", type=str,   default="./weights/autoencoder")
    args = parser.parse_args()
    train(args)
