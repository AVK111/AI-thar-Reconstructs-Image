"""
train.py — GAN training loop for image inpainting (CPU-friendly version)

Usage:
    # Quick smoke test (50 images, 2 epochs)
    python train.py --dataset combined --epochs 2 --batch_size 2 --max_images 50 --save_every 1

    # Proper CPU training (500 images, 10 epochs — takes ~1-2 hrs)
    python train.py --dataset combined --epochs 10 --batch_size 2 --max_images 500 --save_every 2

    # When GPU is available (full dataset)
    python train.py --dataset combined --epochs 100 --batch_size 8 --save_every 5
"""

import argparse
import os
import time
from pathlib import Path

import tensorflow as tf

from dataset       import make_combined_dataset, make_celeba_dataset, make_imagenet_dataset
from generator     import build_generator
from discriminator import build_global_discriminator, build_local_discriminator, extract_masked_crop
from losses        import discriminator_loss, generator_total_loss


# ──────────────────────────────────────────────
# Device setup
# ──────────────────────────────────────────────


def configure_device():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[train] {len(gpus)} GPU(s) detected and configured.")
    else:
        print("[train] Running on CPU.")
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(4)



# ──────────────────────────────────────────────
# Metrics — PSNR and SSIM
# ──────────────────────────────────────────────

def compute_psnr(real: tf.Tensor, fake: tf.Tensor) -> float:
    """
    Peak Signal-to-Noise Ratio between real and generated images.
    Converts [-1,1] → [0,1] before computing.
    Higher is better. Typical good inpainting: 25–35 dB.
    """
    real_01 = (real + 1.0) / 2.0
    fake_01 = (fake + 1.0) / 2.0
    psnr    = tf.image.psnr(real_01, fake_01, max_val=1.0)
    return float(tf.reduce_mean(psnr))


def compute_ssim(real: tf.Tensor, fake: tf.Tensor) -> float:
    """
    Structural Similarity Index between real and generated images.
    Converts [-1,1] → [0,1] before computing.
    Higher is better. Range: 0–1. Good inpainting: > 0.7.
    """
    real_01 = (real + 1.0) / 2.0
    fake_01 = (fake + 1.0) / 2.0
    ssim    = tf.image.ssim(real_01, fake_01, max_val=1.0)
    return float(tf.reduce_mean(ssim))


# ──────────────────────────────────────────────
# Single training step
# ──────────────────────────────────────────────

@tf.function
def train_step(
    masked_image, mask, real_image,
    generator, global_disc, local_disc,
    gen_opt, global_disc_opt, local_disc_opt,
):
    with tf.GradientTape() as gen_tape, \
         tf.GradientTape() as gdisc_tape, \
         tf.GradientTape() as ldisc_tape:

        # Cast all inputs to float32 to avoid float16/float32 mismatch with mixed precision
        masked_image = tf.cast(masked_image, tf.float32)
        mask         = tf.cast(mask,         tf.float32)
        real_image   = tf.cast(real_image,   tf.float32)

        gen_input  = tf.concat([masked_image, mask], axis=-1)
        fake_image = tf.cast(generator(gen_input, training=True), tf.float32)
        composite  = masked_image * (1.0 - mask) + fake_image * mask

        real_crop  = extract_masked_crop(real_image, mask)
        fake_crop  = extract_masked_crop(composite,  mask)

        real_global = tf.cast(global_disc([real_image, mask], training=True), tf.float32)
        fake_global = tf.cast(global_disc([composite,  mask], training=True), tf.float32)
        real_local  = tf.cast(local_disc(real_crop, training=True), tf.float32)
        fake_local  = tf.cast(local_disc(fake_crop, training=True), tf.float32)

        g_losses      = generator_total_loss(real_image, composite, mask, fake_global, fake_local)
        d_global_loss = discriminator_loss(real_global, fake_global)
        d_local_loss  = discriminator_loss(real_local,  fake_local)

    gen_grads   = gen_tape.gradient(g_losses["total"], generator.trainable_variables)
    gdisc_grads = gdisc_tape.gradient(d_global_loss,   global_disc.trainable_variables)
    ldisc_grads = ldisc_tape.gradient(d_local_loss,    local_disc.trainable_variables)

    gen_opt.apply_gradients(        zip(gen_grads,   generator.trainable_variables))
    global_disc_opt.apply_gradients(zip(gdisc_grads, global_disc.trainable_variables))
    local_disc_opt.apply_gradients( zip(ldisc_grads, local_disc.trainable_variables))

    return {
        "g_total":      g_losses["total"],
        "g_l1":         g_losses["l1"],
        "g_adv":        g_losses["adv_global"] + g_losses["adv_local"],
        "g_perceptual": g_losses["perceptual"],
        "g_style":      g_losses["style"],
        "d_global":     d_global_loss,
        "d_local":      d_local_loss,
        "composite":    composite,   # returned for metric computation
    }


# ──────────────────────────────────────────────
# Sample image saver
# ──────────────────────────────────────────────

def save_samples(generator, sample_batch, epoch, out_dir):
    masked_image, mask, real_image = sample_batch
    n          = min(4, masked_image.shape[0])
    gen_input  = tf.concat([masked_image[:n], mask[:n]], axis=-1)
    fake       = generator(gen_input, training=False)
    composite  = masked_image[:n] * (1.0 - mask[:n]) + fake * mask[:n]

    def to_uint8(t):
        return tf.cast(tf.clip_by_value((t + 1.0) * 127.5, 0, 255), tf.uint8)

    # Grid: masked | inpainted | original  (side by side per image)
    row  = tf.concat([to_uint8(masked_image[:n]), to_uint8(composite), to_uint8(real_image[:n])], axis=2)
    grid = tf.concat(tf.unstack(row, axis=0), axis=0)
    path = os.path.join(out_dir, f"epoch_{epoch:04d}.png")
    tf.io.write_file(path, tf.image.encode_png(grid))
    print(f"  [sample] saved → {path}")

    # Return composite for metric display on sample batch
    return composite, real_image[:n]


# ──────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────

def train(args):
    configure_device()

    # ── Dataset ──────────────────────────────
    if args.dataset == "combined":
        dataset = make_combined_dataset(
            batch_size=args.batch_size,
            celeba_ratio=args.celeba_ratio,
            celeba_csv=args.celeba_csv,
            imagenet_dir=args.imagenet_dir,
        )
    elif args.dataset == "celeba":
        dataset = make_celeba_dataset(batch_size=args.batch_size, csv_path=args.celeba_csv)
    else:
        dataset = make_imagenet_dataset(batch_size=args.batch_size, imagenet_dir=args.imagenet_dir)

    # Limit dataset size for CPU training
    if args.max_images > 0:
        steps_limit = max(1, args.max_images // args.batch_size)
        dataset     = dataset.take(steps_limit)
        print(f"[train] CPU mode: using {steps_limit} steps ({args.max_images} images) per epoch")

    sample_batch = next(iter(dataset))

    # ── Models ───────────────────────────────
    generator   = build_generator()
    global_disc = build_global_discriminator()
    local_disc  = build_local_discriminator()

    # ── Optimisers ───────────────────────────
    gen_opt         = tf.keras.optimizers.Adam(args.lr,       beta_1=0.5)
    global_disc_opt = tf.keras.optimizers.Adam(args.lr * 0.5, beta_1=0.5)
    local_disc_opt  = tf.keras.optimizers.Adam(args.lr * 0.5, beta_1=0.5)

    # ── Checkpoint ───────────────────────────
    ckpt_dir   = Path(args.checkpoint_dir)
    sample_dir = ckpt_dir / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(exist_ok=True)

    checkpoint = tf.train.Checkpoint(
        generator=generator, global_disc=global_disc, local_disc=local_disc,
        gen_opt=gen_opt, global_disc_opt=global_disc_opt, local_disc_opt=local_disc_opt,
    )
    ckpt_manager = tf.train.CheckpointManager(checkpoint, str(ckpt_dir), max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print(f"[train] Restored checkpoint: {ckpt_manager.latest_checkpoint}")

    # ── TensorBoard ──────────────────────────
    writer = tf.summary.create_file_writer(str(ckpt_dir / "logs"))

    # ── Training ─────────────────────────────
    print(f"\n[train] Starting training for {args.epochs} epochs\n")
    print(f"{'Epoch':>6} | {'G_loss':>8} {'L1':>8} {'Adv':>8} {'Perc':>8} | "
          f"{'D_g':>8} {'D_l':>8} | {'PSNR':>7} {'SSIM':>6} | {'Time':>7}")
    print("─" * 95)

    best_psnr = 0.0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        totals = {k: 0.0 for k in ["g_total","g_l1","g_adv","g_perceptual","g_style","d_global","d_local"]}
        psnr_list, ssim_list = [], []
        steps = 0

        for masked_image, mask, real_image in dataset:
            result = train_step(
                masked_image, mask, real_image,
                generator, global_disc, local_disc,
                gen_opt, global_disc_opt, local_disc_opt,
            )
            composite = result.pop("composite")
            for k in totals:
                totals[k] += float(result[k])

            # Compute PSNR & SSIM every step
            psnr_list.append(compute_psnr(real_image, composite))
            ssim_list.append(compute_ssim(real_image, composite))
            steps += 1

            # Show step progress for CPU (slow) runs
            if steps % 10 == 0:
                print(f"  step {steps:4d} | "
                      f"G={totals['g_total']/steps:.3f} | "
                      f"PSNR={sum(psnr_list)/len(psnr_list):.2f}dB | "
                      f"SSIM={sum(ssim_list)/len(ssim_list):.4f}", end="\r")

        avg     = {k: totals[k] / max(steps, 1) for k in totals}
        avg_psnr = sum(psnr_list) / max(len(psnr_list), 1)
        avg_ssim = sum(ssim_list) / max(len(ssim_list), 1)
        elapsed = time.time() - epoch_start

        print(f"\r{epoch:6d} | "
              f"{avg['g_total']:8.4f} {avg['g_l1']:8.4f} {avg['g_adv']:8.4f} {avg['g_perceptual']:8.4f} | "
              f"{avg['d_global']:8.4f} {avg['d_local']:8.4f} | "
              f"{avg_psnr:7.2f} {avg_ssim:6.4f} | "
              f"{elapsed:6.1f}s")

        # TensorBoard logging
        with writer.as_default():
            for k, v in avg.items():
                tf.summary.scalar(f"loss/{k}", v, step=epoch)
            tf.summary.scalar("metrics/psnr", avg_psnr, step=epoch)
            tf.summary.scalar("metrics/ssim", avg_ssim, step=epoch)

        # Save samples + checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            comp, real = save_samples(generator, sample_batch, epoch, str(sample_dir))
            ckpt_manager.save()
            print(f"  [ckpt] saved at epoch {epoch}")

        # Save best model by PSNR
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            generator.save(str(ckpt_dir / "generator_best.keras"))
            print(f"  [best] new best PSNR={best_psnr:.2f}dB → generator_best.keras saved")

    print("\n[train] Training complete.")
    generator.save(str(ckpt_dir / "generator_final.keras"))
    print(f"[train] Final model → {ckpt_dir / 'generator_final.keras'}")
    print(f"[train] Best PSNR achieved: {best_psnr:.2f} dB")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAN inpainting model")
    parser.add_argument("--dataset",         type=str,   default="combined",
                        choices=["combined", "celeba", "imagenet"])
    parser.add_argument("--celeba_csv",      type=str,   default=None)
    parser.add_argument("--imagenet_dir",    type=str,   default=None)
    parser.add_argument("--celeba_ratio",    type=float, default=0.5)
    parser.add_argument("--epochs",          type=int,   default=10)
    parser.add_argument("--batch_size",      type=int,   default=2)
    parser.add_argument("--lr",              type=float, default=2e-4)
    parser.add_argument("--save_every",      type=int,   default=2)
    parser.add_argument("--max_images",      type=int,   default=200,
                        help="Max images per epoch for CPU training. Set 0 for full dataset.")
    parser.add_argument("--checkpoint_dir",  type=str,   default="./weights/gan")
    args = parser.parse_args()
    train(args)
