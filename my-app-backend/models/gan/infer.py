"""
infer.py — Load trained GAN generator and inpaint an image.

Usage:
    python infer.py \
        --model_path  ./weights/gan/generator_final.keras \
        --image_path  ./test_image.jpg \
        --mask_path   ./test_mask.png \
        --output_path ./result.png
"""

import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path


IMAGE_SIZE = 256


# ──────────────────────────────────────────────
# Image I/O helpers
# ──────────────────────────────────────────────

def load_image(path: str) -> tf.Tensor:
    """Load and normalise an RGB image to [-1, 1], shape (1, 256, 256, 3)."""
    raw   = tf.io.read_file(path)
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return tf.expand_dims(image, axis=0)             # (1,256,256,3)


def load_mask(path: str) -> tf.Tensor:
    """
    Load a grayscale mask image, binarise (threshold 0.5), shape (1, 256, 256, 1).
    White pixels (255) = missing region to inpaint.
    """
    raw  = tf.io.read_file(path)
    mask = tf.image.decode_image(raw, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, [IMAGE_SIZE, IMAGE_SIZE])
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.cast(mask > 0.5, tf.float32)           # binarise
    return tf.expand_dims(mask, axis=0)              # (1,256,256,1)


def save_image(tensor: tf.Tensor, path: str):
    """Save a (1, H, W, 3) float tensor in [-1, 1] as a PNG file."""
    img = tf.squeeze(tensor, axis=0)                 # (H,W,3)
    img = tf.cast((img + 1.0) * 127.5, tf.uint8)
    encoded = tf.image.encode_png(img)
    tf.io.write_file(path, encoded)
    print(f"[infer] Saved → {path}")


# ──────────────────────────────────────────────
# Inpainting function
# ──────────────────────────────────────────────

def inpaint(
    generator:    tf.keras.Model,
    image:        tf.Tensor,
    mask:         tf.Tensor,
    blend:        bool = True,
) -> tf.Tensor:
    """
    Run inpainting on a single image.

    Args:
        generator : loaded Keras generator model
        image     : (1, 256, 256, 3) in [-1, 1]
        mask      : (1, 256, 256, 1)  1=missing region
        blend     : if True, paste generated pixels ONLY into masked region
                    (keeps known pixels pixel-perfect from original)

    Returns:
        result : (1, 256, 256, 3) in [-1, 1]
    """
    masked_image = image * (1.0 - mask)              # zero out missing region
    gen_input    = tf.concat([masked_image, mask], axis=-1)  # (1,256,256,4)
    prediction   = generator(gen_input, training=False)      # (1,256,256,3)

    if blend:
        result = masked_image * (1.0 - mask) + prediction * mask
    else:
        result = prediction

    return result


# ──────────────────────────────────────────────
# Batch inference helper (for API use)
# ──────────────────────────────────────────────

def inpaint_from_numpy(
    generator:    tf.keras.Model,
    image_np:     np.ndarray,
    mask_np:      np.ndarray,
) -> np.ndarray:
    """
    Inpaint from numpy arrays — for FastAPI service use.
    Applies Gaussian smoothing inside the hole region post-inference.
    """
    from scipy.ndimage import gaussian_filter

    orig_h, orig_w = image_np.shape[:2]

    image_tf = tf.image.resize(image_np[np.newaxis].astype(np.float32), [IMAGE_SIZE, IMAGE_SIZE])
    image_tf = image_tf / 127.5 - 1.0

    mask_3d = mask_np[:, :, np.newaxis].astype(np.float32) / 255.0
    mask_tf = tf.image.resize(mask_3d[np.newaxis], [IMAGE_SIZE, IMAGE_SIZE])
    mask_tf = tf.cast(mask_tf > 0.5, tf.float32)

    result      = inpaint(generator, image_tf, mask_tf)
    result_u8   = tf.cast(tf.clip_by_value((result + 1.0) * 127.5, 0, 255), tf.uint8)
    result_orig = tf.image.resize(result_u8, [orig_h, orig_w]).numpy().squeeze(0).astype(np.float32)

    # Gaussian smoothing inside hole only
    smoothed    = gaussian_filter(result_orig, sigma=[0.6, 0.6, 0])
    mask_res    = tf.image.resize(mask_3d[np.newaxis], [orig_h, orig_w]).numpy().squeeze(0)
    mask_bin    = (mask_res > 0.5).astype(np.float32)
    result_orig = result_orig * (1.0 - mask_bin) + smoothed * mask_bin

    return np.clip(result_orig, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def main(args):
    print(f"[infer] Loading model from {args.model_path} …")
    generator = tf.keras.models.load_model(args.model_path)
    print("[infer] Model loaded.")

    image = load_image(args.image_path)
    mask  = load_mask(args.mask_path)

    print("[infer] Running inpainting …")
    result = inpaint(generator, image, mask, blend=True)

    save_image(result, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN image inpainting inference")
    parser.add_argument("--model_path",  required=True, help="Path to generator_final.keras")
    parser.add_argument("--image_path",  required=True, help="Input image path (jpg/png)")
    parser.add_argument("--mask_path",   required=True, help="Mask image path (white=inpaint)")
    parser.add_argument("--output_path", default="./inpainted_result.png")
    args = parser.parse_args()
    main(args)
