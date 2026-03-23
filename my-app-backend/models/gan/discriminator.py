import tensorflow as tf
from tensorflow.keras import layers, Model


# ──────────────────────────────────────────────
# PatchGAN Discriminator
# ──────────────────────────────────────────────
# PatchGAN classifies overlapping N×N patches as real/fake rather than
# the whole image. This enforces local texture realism — exactly what
# inpainting needs. We use two discriminators:
#   - Global : sees the full 256×256 image (overall coherence)
#   - Local  : sees only the 128×128 masked region crop (fine detail)

def _disc_block(x, filters, strides=2, use_bn=True):
    """Conv → (BN) → LeakyReLU."""
    x = layers.Conv2D(
        filters,
        kernel_size=4,
        strides=strides,
        padding="same",
        use_bias=not use_bn,
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    return x


def build_global_discriminator(image_size: int = 256) -> Model:
    """
    Global PatchGAN discriminator.

    Input  : real or fake RGB image  (B, 256, 256, 3)
             concatenated with mask   (B, 256, 256, 1)  → 4 channels total
    Output : patch validity map      (B, 16, 16, 1)
             values in (0,1) via sigmoid — 1=real, 0=fake
    """
    img  = layers.Input(shape=(image_size, image_size, 3), name="image")
    mask = layers.Input(shape=(image_size, image_size, 1), name="mask")

    x = layers.Concatenate()([img, mask])        # (256,256,4)

    x = _disc_block(x, 32,  use_bn=False)        # (128,128, 32)
    x = _disc_block(x, 64)                       # (64, 64,  64)
    x = _disc_block(x, 128)                      # (32, 32, 128)
    x = _disc_block(x, 256)                      # (16, 16, 256)

    # Final conv → patch validity map (no activation — applied in loss)
    out = layers.Conv2D(
        1, kernel_size=4,
        strides=1,
        padding="same",
        name="patch_validity",
    )(x)                                          # (16, 16, 1)

    return Model(inputs=[img, mask], outputs=out, name="Global_Discriminator")


def build_local_discriminator(crop_size: int = 128) -> Model:
    """
    Local PatchGAN discriminator — focuses on the damaged region crop.

    Input  : cropped region of real or fake image  (B, 128, 128, 3)
    Output : patch validity map                    (B, 8, 8, 1)
    """
    inp = layers.Input(shape=(crop_size, crop_size, 3), name="region_crop")

    x = _disc_block(inp, 32,  use_bn=False)       # (64, 64,  32)
    x = _disc_block(x,   64)                      # (32, 32,  64)
    x = _disc_block(x,   128)                     # (16, 16, 128)
    x = _disc_block(x,   256)                     # (8,  8,  256)

    out = layers.Conv2D(
        1, kernel_size=4,
        strides=1,
        padding="same",
        name="local_patch_validity",
    )(x)                                           # (8, 8, 1)

    return Model(inputs=inp, outputs=out, name="Local_Discriminator")


# ──────────────────────────────────────────────
# Crop helper for local discriminator
# ──────────────────────────────────────────────

def extract_masked_crop(
    image: tf.Tensor,
    mask:  tf.Tensor,
    crop_size: int = 128,
) -> tf.Tensor:
    """
    Extract a fixed-size crop centred on the masked region bounding box.
    Falls back to a random crop if no mask pixels exist.

    Args:
        image     : (B, H, W, 3)
        mask      : (B, H, W, 1)  1=masked region
        crop_size : height == width of output crop

    Returns:
        crops : (B, crop_size, crop_size, 3)
    """
    batch_size = tf.shape(image)[0]
    h          = tf.shape(image)[1]
    w          = tf.shape(image)[2]

    def crop_one(args):
        img_i, mask_i = args                           # (H,W,3), (H,W,1)
        mask_2d = tf.squeeze(mask_i, axis=-1)          # (H,W)

        # Bounding box of masked region
        rows = tf.where(tf.reduce_any(mask_2d > 0.5, axis=1))
        cols = tf.where(tf.reduce_any(mask_2d > 0.5, axis=0))

        def bbox_crop():
            r_min = tf.cast(tf.reduce_min(rows), tf.int32)
            r_max = tf.cast(tf.reduce_max(rows), tf.int32)
            c_min = tf.cast(tf.reduce_min(cols), tf.int32)
            c_max = tf.cast(tf.reduce_max(cols), tf.int32)
            cy    = (r_min + r_max) // 2
            cx    = (c_min + c_max) // 2
            y0    = tf.clip_by_value(cy - crop_size // 2, 0, h - crop_size)
            x0    = tf.clip_by_value(cx - crop_size // 2, 0, w - crop_size)
            return img_i[y0:y0+crop_size, x0:x0+crop_size, :]

        def rand_crop():
            y0 = tf.random.uniform((), 0, h - crop_size, dtype=tf.int32)
            x0 = tf.random.uniform((), 0, w - crop_size, dtype=tf.int32)
            return img_i[y0:y0+crop_size, x0:x0+crop_size, :]

        has_mask = tf.size(rows) > 0
        return tf.cond(has_mask, bbox_crop, rand_crop)

    crops = tf.map_fn(
        crop_one,
        (image, mask),
        fn_output_signature=tf.TensorSpec([crop_size, crop_size, 3], tf.float32),
    )
    return crops                                      # (B, crop_size, crop_size, 3)


# ──────────────────────────────────────────────
# Quick check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    global_disc = build_global_discriminator()
    local_disc  = build_local_discriminator()

    global_disc.summary()
    local_disc.summary()

    fake_img  = tf.random.normal((2, 256, 256, 3))
    fake_mask = tf.random.uniform((2, 256, 256, 1), 0, 1)

    g_out = global_disc([fake_img, fake_mask], training=False)
    print("\nGlobal disc output shape:", g_out.shape)  # (2, 16, 16, 1)

    crop = extract_masked_crop(fake_img, fake_mask)
    l_out = local_disc(crop, training=False)
    print("Local  disc output shape:", l_out.shape)   # (2, 8, 8, 1)
