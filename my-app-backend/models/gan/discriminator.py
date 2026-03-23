import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
IMAGE_SIZE = 256
AUTOTUNE   = tf.data.AUTOTUNE

# ──────────────────────────────────────────────
# Project folder layout
#
#   <project_root>/
#       Datasets/
#           CelebA/
#               train.csv          ← rows: (path, label)
#               img_align_celeba/  ← actual images in subfolders
#           ImageNET/
#               train.csv
#               <synset_folders>/  ← actual images in subfolders
#       backend/
#       frontend/
#
# DATASETS_ROOT is resolved relative to this file's location:
#   backend/models/gan/dataset.py  →  ../../..  →  project root
# ──────────────────────────────────────────────

_THIS_FILE    = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]          # backend/models/gan → project root
DATASETS_ROOT = _PROJECT_ROOT / "Datasets"

CELEBA_DIR   = DATASETS_ROOT / "CelebA"
IMAGENET_DIR = DATASETS_ROOT / "ImageNET"

# CSV column names — matched to your actual dataset CSVs
# CelebA  : image_id  (e.g. 000001.jpg)
# ImageNet : check your CSV header — update IMAGENET_PATH_COL if different
CELEBA_PATH_COL   = "image_id"   # CelebA:  list_attr_celeba.csv
IMAGENET_PATH_COL = "path"       # ImageNet: update if your column differs


# ──────────────────────────────────────────────
# CSV reader
# ──────────────────────────────────────────────

def load_paths_from_csv(csv_path: Path, dataset_root: Path, path_col: str = "path") -> list[str]:
    """
    Read a train CSV and return a list of absolute image paths.

    Builds a filename→path index ONCE by scanning dataset_root, then resolves
    every CSV row against that index in O(1). Avoids per-row rglob which is
    extremely slow for large datasets like CelebA (200k+ images).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if path_col not in df.columns:
        raise ValueError(
            f"Expected a '{path_col}' column in {csv_path}. "
            f"Found columns: {list(df.columns)}"
        )

    # Build filename → absolute path index in one pass (fast)
    print(f"[dataset] Scanning {dataset_root} to build path index …")
    exts  = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    index = {}
    for p in dataset_root.rglob("*"):
        if p.suffix in exts:
            index[p.name] = str(p)           # e.g. "000001.jpg" → full path
    print(f"[dataset] Index built: {len(index):,} images found on disk")

    # Resolve CSV rows against the index
    abs_paths = []
    missing   = 0

    for raw_path in df[path_col].astype(str):
        p = Path(raw_path)

        # Absolute path that exists → use directly
        if p.is_absolute() and p.exists():
            abs_paths.append(str(p))
            continue

        # Relative path resolves directly under dataset_root
        resolved = dataset_root / p
        if resolved.exists():
            abs_paths.append(str(resolved))
            continue

        # Filename-only lookup in the pre-built index (covers CelebA subfolders)
        if p.name in index:
            abs_paths.append(index[p.name])
        else:
            missing += 1

    print(f"[dataset] {csv_path.name}: {len(abs_paths):,} valid paths "
          f"({missing} missing skipped)")

    if not abs_paths:
        raise FileNotFoundError(
            f"No valid image paths found in {csv_path}. "
            f"Check that path_col='{path_col}' matches your CSV header "
            f"and that images exist under {dataset_root}."
        )

    return abs_paths


def scan_image_dir(dataset_root: Path) -> list[str]:
    """
    Recursively collect all .jpg / .png image paths under dataset_root.

    Designed for ImageNet's structure:
        ImageNET/
            abacus/
                000.jpg
                001.jpg
            accordion/
                000.jpg
            ...

    Returns a list of absolute path strings. No CSV needed.
    """
    exts  = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    paths = [str(p) for p in dataset_root.rglob("*") if p.suffix in exts]

    if not paths:
        raise FileNotFoundError(
            f"No images found under {dataset_root}. "
            f"Expected subfolders containing .jpg files."
        )

    print(f"[dataset] ImageNet: found {len(paths):,} images across "
          f"{len(set(p.parent for p in map(Path, paths)))} class folders")
    return paths


def find_csv(dataset_dir: Path) -> Path:
    """
    Auto-discover the train CSV inside a dataset folder.
    Looks for files matching train*.csv (case-insensitive).
    Falls back to the first .csv found.
    Raises if none found.
    """
    candidates = sorted(dataset_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    # Prefer files whose name starts with 'train'
    train_csvs = [c for c in candidates if c.stem.lower().startswith("train")]
    chosen = train_csvs[0] if train_csvs else candidates[0]
    print(f"[dataset] Using CSV: {chosen}")
    return chosen


# ──────────────────────────────────────────────
# Image loading & preprocessing
# ──────────────────────────────────────────────

def load_and_preprocess(path: tf.Tensor) -> tf.Tensor:
    """Load a JPEG/PNG from an absolute path, resize to 256×256, normalise to [-1, 1]."""
    raw   = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=3)   # decode_jpeg handles PNG too
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return image


def random_augment(image: tf.Tensor) -> tf.Tensor:
    """Horizontal flip only — preserves face structure for CelebA."""
    return tf.image.random_flip_left_right(image)


# ──────────────────────────────────────────────
# Free-form mask generation
# ──────────────────────────────────────────────

def generate_freeform_mask(
    height: int = IMAGE_SIZE,
    width:  int = IMAGE_SIZE,
    num_strokes: int = 5,
    max_len:     int = 80,
    max_width:   int = 25,
) -> np.ndarray:
    """
    Simulate a user brush stroke over a damaged region.

    Returns float32 (H, W, 1):  1 = missing/inpaint here,  0 = known pixel.
    """
    mask = np.zeros((height, width), dtype=np.float32)

    for _ in range(num_strokes):
        x, y         = np.random.randint(0, width), np.random.randint(0, height)
        angle        = np.random.uniform(0, 2 * np.pi)
        length       = np.random.randint(10, max_len)
        stroke_width = np.random.randint(8, max_width)

        for _ in range(length):
            r0 = max(0, y - stroke_width // 2)
            r1 = min(height, y + stroke_width // 2)
            c0 = max(0, x - stroke_width // 2)
            c1 = min(width,  x + stroke_width // 2)
            mask[r0:r1, c0:c1] = 1.0

            angle += np.random.uniform(-0.4, 0.4)
            x = int(np.clip(x + np.cos(angle) * 4, 0, width  - 1))
            y = int(np.clip(y + np.sin(angle) * 4, 0, height - 1))

    return mask[:, :, np.newaxis]   # (H, W, 1)


def tf_generate_mask(_image: tf.Tensor) -> tf.Tensor:
    """tf.data-compatible wrapper around generate_freeform_mask."""
    mask = tf.numpy_function(
        func=lambda: generate_freeform_mask(),
        inp=[],
        Tout=tf.float32,
    )
    mask.set_shape([IMAGE_SIZE, IMAGE_SIZE, 1])
    return mask


# ──────────────────────────────────────────────
# Core pipeline builder
# ──────────────────────────────────────────────

def _build_pipeline(
    abs_paths:  list[str],
    batch_size: int  = 8,
    augment:    bool = True,
    shuffle:    bool = True,
    cache:      bool = False,
) -> tf.data.Dataset:
    """
    Internal helper: build a tf.data pipeline from a list of absolute paths.

    Yields batches of (masked_image, mask, original_image):
        masked_image : (B, 256, 256, 3)  original with hole zeroed out
        mask         : (B, 256, 256, 1)  1=hole  0=known
        original     : (B, 256, 256, 3)  ground truth
    """
    ds = tf.data.Dataset.from_tensor_slices(abs_paths)

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(abs_paths), 10_000),
            reshuffle_each_iteration=True,
        )

    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)

    if augment:
        ds = ds.map(random_augment, num_parallel_calls=AUTOTUNE)

    if cache:
        ds = ds.cache()

    def attach_mask(image: tf.Tensor):
        mask         = tf_generate_mask(image)
        masked_image = image * (1.0 - mask)
        return masked_image, mask, image

    ds = ds.map(attach_mask, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ──────────────────────────────────────────────
# Public dataset builders
# ──────────────────────────────────────────────

def make_celeba_dataset(
    batch_size: int  = 8,
    augment:    bool = True,
    shuffle:    bool = True,
    cache:      bool = False,
    csv_path:   str  = None,
) -> tf.data.Dataset:
    """
    Load the CelebA dataset from:
        Datasets/CelebA/train.csv

    Args:
        csv_path : override the auto-discovered CSV path (optional).
    """
    csv   = Path(csv_path) if csv_path else find_csv(CELEBA_DIR)
    paths = load_paths_from_csv(csv, CELEBA_DIR, path_col=CELEBA_PATH_COL)
    print(f"[dataset] CelebA → {len(paths):,} images")
    return _build_pipeline(paths, batch_size, augment, shuffle, cache)


def make_imagenet_dataset(
    batch_size: int  = 8,
    augment:    bool = True,
    shuffle:    bool = True,
    cache:      bool = False,
    imagenet_dir: str = None,
) -> tf.data.Dataset:
    """
    Load ImageNet directly from its class subfolders — no CSV needed.

    Folder structure expected:
        Datasets/ImageNET/
            abacus/   000.jpg  001.jpg ...
            accordion/ 000.jpg ...
            ...

    Args:
        imagenet_dir : override the default Datasets/ImageNET path (optional).
    """
    root  = Path(imagenet_dir) if imagenet_dir else IMAGENET_DIR
    paths = scan_image_dir(root)
    return _build_pipeline(paths, batch_size, augment, shuffle, cache)


def make_combined_dataset(
    batch_size:   int   = 8,
    celeba_ratio: float = 0.5,
    augment:      bool  = True,
    celeba_csv:   str   = None,
    imagenet_dir: str   = None,
) -> tf.data.Dataset:
    """
    Interleave CelebA (CSV-based) and ImageNet (folder-based) into one dataset.

    Args:
        celeba_ratio : fraction of each batch from CelebA (0–1).
        celeba_csv   : override path to CelebA CSV  (optional).
        imagenet_dir : override path to ImageNet root folder (optional).
    """
    celeba_bs   = max(1, int(batch_size * celeba_ratio))
    imagenet_bs = max(1, batch_size - celeba_bs)

    ds_celeba   = make_celeba_dataset(  batch_size=celeba_bs,   augment=augment, csv_path=celeba_csv)
    ds_imagenet = make_imagenet_dataset(batch_size=imagenet_bs, augment=augment, imagenet_dir=imagenet_dir)

    combined = tf.data.Dataset.zip((ds_celeba, ds_imagenet))

    def merge(celeba_batch, imagenet_batch):
        return tuple(
            tf.concat([c, i], axis=0)
            for c, i in zip(celeba_batch, imagenet_batch)
        )

    combined = combined.map(merge, num_parallel_calls=AUTOTUNE)
    combined = combined.prefetch(AUTOTUNE)

    print(f"[dataset] Combined: {celeba_bs} CelebA + {imagenet_bs} ImageNet per batch")
    return combined


# ──────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("── CelebA ──────────────────────────")
    ds_c = make_celeba_dataset(batch_size=4)
    for masked, mask, original in ds_c.take(1):
        print("  masked  :", masked.shape,   masked.numpy().min(),   masked.numpy().max())
        print("  mask    :", mask.shape,     mask.numpy().min(),     mask.numpy().max())
        print("  original:", original.shape, original.numpy().min(), original.numpy().max())

    print("\n── ImageNet ────────────────────────")
    ds_i = make_imagenet_dataset(batch_size=4)
    for masked, mask, original in ds_i.take(1):
        print("  masked  :", masked.shape)
        print("  original:", original.shape)

    print("\n── Combined ────────────────────────")
    ds_comb = make_combined_dataset(batch_size=8, celeba_ratio=0.5)
    for masked, mask, original in ds_comb.take(1):
        print("  combined batch:", masked.shape)   # (8, 256, 256, 3)

    print("\nAll pipelines OK.")
