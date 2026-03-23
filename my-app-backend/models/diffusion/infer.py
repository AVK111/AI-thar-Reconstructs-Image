"""
infer.py — Load trained Diffusion U-Net and inpaint an image.

Usage:
    python infer.py \
        --model_path  ./weights/diffusion/unet_best.keras \
        --image_path  ./test_image.jpg \
        --mask_path   ./test_mask.png \
        --output_path ./result.png \
        --steps       100
"""

import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

from scheduler import DDPMScheduler

IMAGE_SIZE = 256


def load_image(path: str) -> tf.Tensor:
    raw   = tf.io.read_file(path)
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return tf.expand_dims(image, 0)


def load_mask(path: str) -> tf.Tensor:
    raw  = tf.io.read_file(path)
    mask = tf.image.decode_image(raw, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, [IMAGE_SIZE, IMAGE_SIZE])
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.cast(mask > 0.5, tf.float32)
    return tf.expand_dims(mask, 0)


def save_image(tensor: tf.Tensor, path: str):
    img     = tf.squeeze(tensor, 0)
    img     = tf.cast(tf.clip_by_value((img + 1.0) * 127.5, 0, 255), tf.uint8)
    encoded = tf.image.encode_png(img)
    tf.io.write_file(path, encoded)
    print(f"[infer] Saved → {path}")


def inpaint_from_numpy(
    unet:       tf.keras.Model,
    scheduler:  DDPMScheduler,
    image_np:   np.ndarray,
    mask_np:    np.ndarray,
    steps:      int = 50,
) -> np.ndarray:
    """
    Inpaint from numpy arrays — for FastAPI service use.

    Args:
        image_np : uint8 (H, W, 3)
        mask_np  : uint8 (H, W)    white = region to inpaint
        steps    : denoising steps (50 is fast, 200 is better quality)

    Returns:
        uint8 (H, W, 3)
    """
    orig_h, orig_w = image_np.shape[:2]

    image_tf = tf.image.resize(image_np[np.newaxis].astype(np.float32), [IMAGE_SIZE, IMAGE_SIZE])
    image_tf = image_tf / 127.5 - 1.0

    mask_3d = mask_np[:, :, np.newaxis].astype(np.float32) / 255.0
    mask_tf = tf.image.resize(mask_3d[np.newaxis], [IMAGE_SIZE, IMAGE_SIZE])
    mask_tf = tf.cast(mask_tf > 0.5, tf.float32)

    # Run reverse diffusion
    batch_size = 1
    x_t  = tf.random.normal(tf.shape(image_tf))
    x_t  = image_tf * (1.0 - mask_tf) + x_t * mask_tf

    step_size = scheduler.T // steps
    timesteps = list(range(scheduler.T - 1, -1, -step_size))

    for t in timesteps:
        t_batch    = tf.fill([batch_size], t)
        noise_pred = unet([x_t, mask_tf, t_batch], training=False)
        x_t        = scheduler.step(x_t, t, noise_pred, image_tf, mask_tf)

    result      = tf.clip_by_value(x_t, -1.0, 1.0)
    result_u8   = tf.cast((result + 1.0) * 127.5, tf.uint8)
    result_orig = tf.image.resize(result_u8, [orig_h, orig_w])
    return result_orig.numpy().squeeze(0).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  required=True)
    parser.add_argument("--image_path",  required=True)
    parser.add_argument("--mask_path",   required=True)
    parser.add_argument("--output_path", default="./diffusion_inpainted.png")
    parser.add_argument("--steps",       type=int, default=100)
    parser.add_argument("--T",           type=int, default=1000)
    args = parser.parse_args()

    print(f"[infer] Loading U-Net from {args.model_path} …")
    unet      = tf.keras.models.load_model(args.model_path, compile=False)
    scheduler = DDPMScheduler(T=args.T)
    print("[infer] Model loaded. Running diffusion …")

    image  = load_image(args.image_path)
    mask   = load_mask(args.mask_path)

    # Run inference
    x_t = tf.random.normal(tf.shape(image))
    x_t = image * (1.0 - mask) + x_t * mask

    step_size = scheduler.T // args.steps
    timesteps = list(range(scheduler.T - 1, -1, -step_size))

    for i, t in enumerate(timesteps):
        t_batch    = tf.fill([1], t)
        noise_pred = unet([x_t, mask, t_batch], training=False)
        x_t        = scheduler.step(x_t, t, noise_pred, image, mask)
        if (i + 1) % 10 == 0:
            print(f"  step {i+1}/{len(timesteps)}", end="\r")

    result = tf.clip_by_value(x_t, -1.0, 1.0)
    save_image(result, args.output_path)
