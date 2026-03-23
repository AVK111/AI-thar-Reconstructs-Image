"""
infer.py — Load trained Autoencoder and inpaint an image.

Usage:
    python infer.py \
        --model_path  ./weights/autoencoder/autoencoder_best.keras \
        --image_path  ./test_image.jpg \
        --mask_path   ./test_mask.png \
        --output_path ./result.png
"""

import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

IMAGE_SIZE = 256


def load_image(path: str) -> tf.Tensor:
    raw   = tf.io.read_file(path)
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return tf.expand_dims(image, 0)                  # (1,256,256,3)


def load_mask(path: str) -> tf.Tensor:
    raw  = tf.io.read_file(path)
    mask = tf.image.decode_image(raw, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, [IMAGE_SIZE, IMAGE_SIZE])
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.cast(mask > 0.5, tf.float32)
    return tf.expand_dims(mask, 0)                   # (1,256,256,1)


def save_image(tensor: tf.Tensor, path: str):
    img     = tf.squeeze(tensor, 0)
    img     = tf.cast(tf.clip_by_value((img + 1.0) * 127.5, 0, 255), tf.uint8)
    encoded = tf.image.encode_png(img)
    tf.io.write_file(path, encoded)
    print(f"[infer] Saved → {path}")


def inpaint(autoencoder, image, mask):
    masked   = image * (1.0 - mask)
    ae_input = tf.concat([masked, mask], axis=-1)
    predicted= autoencoder(ae_input, training=False)
    return masked * (1.0 - mask) + predicted * mask


def inpaint_from_numpy(autoencoder, image_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    """
    Inpaint from numpy arrays — for FastAPI service use.

    Args:
        image_np : uint8 (H, W, 3)
        mask_np  : uint8 (H, W)    white = region to inpaint

    Returns:
        uint8 (H, W, 3)
    """
    orig_h, orig_w = image_np.shape[:2]

    image_tf = tf.image.resize(image_np[np.newaxis].astype(np.float32), [IMAGE_SIZE, IMAGE_SIZE])
    image_tf = image_tf / 127.5 - 1.0

    mask_3d = mask_np[:, :, np.newaxis].astype(np.float32) / 255.0
    mask_tf = tf.image.resize(mask_3d[np.newaxis], [IMAGE_SIZE, IMAGE_SIZE])
    mask_tf = tf.cast(mask_tf > 0.5, tf.float32)

    result      = inpaint(autoencoder, image_tf, mask_tf)
    result_u8   = tf.cast(tf.clip_by_value((result + 1.0) * 127.5, 0, 255), tf.uint8)
    result_orig = tf.image.resize(result_u8, [orig_h, orig_w])
    return result_orig.numpy().squeeze(0).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  required=True)
    parser.add_argument("--image_path",  required=True)
    parser.add_argument("--mask_path",   required=True)
    parser.add_argument("--output_path", default="./ae_inpainted.png")
    args = parser.parse_args()

    print(f"[infer] Loading model from {args.model_path} …")
    autoencoder = tf.keras.models.load_model(args.model_path)
    print("[infer] Model loaded.")

    image  = load_image(args.image_path)
    mask   = load_mask(args.mask_path)
    result = inpaint(autoencoder, image, mask)
    save_image(result, args.output_path)
