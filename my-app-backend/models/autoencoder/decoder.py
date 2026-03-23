import tensorflow as tf
from tensorflow.keras import layers, Model


# ──────────────────────────────────────────────
# Decoder
# ──────────────────────────────────────────────

def build_decoder(latent_dim: int = 512, image_size: int = 256) -> Model:
    """
    Convolutional Decoder for image inpainting.

    Takes a latent vector and progressively upsamples it back to a
    full resolution RGB image.

    Input  : (B, latent_dim)   latent representation
    Output : (B, 256, 256, 3)  reconstructed image in [-1, 1]

    Architecture: Dense projection → progressive upsampling
    latent → 4×4×512 → 8 → 16 → 32 → 64 → 128 → 256
    """
    inp = layers.Input(shape=(latent_dim,), name="decoder_input")

    # ── Project latent to spatial ─────────────
    x = layers.Dense(4 * 4 * 512, activation="relu")(inp)
    x = layers.Reshape((4, 4, 512))(x)                        # (4,4,512)

    # ── Upsampling blocks ─────────────────────
    x = layers.Conv2DTranspose(512, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)                                       # (8,8,512)

    x = layers.Conv2DTranspose(512, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)                                       # (16,16,512)

    x = layers.Conv2DTranspose(256, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)                                       # (32,32,256)

    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)                                       # (64,64,128)

    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)                                       # (128,128,64)

    # ── Output head ───────────────────────────
    out = layers.Conv2DTranspose(
        3, 4,
        strides=2,
        padding="same",
        activation="tanh",
        name="reconstructed_image",
    )(x)                                                       # (256,256,3) in [-1,1]

    return Model(inputs=inp, outputs=out, name="Decoder")


# ──────────────────────────────────────────────
# Full Autoencoder (Encoder + Decoder combined)
# ──────────────────────────────────────────────

def build_autoencoder(image_size: int = 256, latent_dim: int = 512) -> Model:
    """
    Full Autoencoder = Encoder + Decoder as a single end-to-end model.

    Input  : (B, 256, 256, 4)  masked image concatenated with mask
    Output : (B, 256, 256, 3)  inpainted image in [-1, 1]

    Used during training. For inference use inpaint_composite() to
    preserve known pixels exactly.
    """
    from encoder import build_encoder

    encoder = build_encoder(image_size, latent_dim)
    decoder = build_decoder(latent_dim, image_size)

    inp     = layers.Input(shape=(image_size, image_size, 4), name="ae_input")
    latent  = encoder(inp)
    out     = decoder(latent)

    return Model(inputs=inp, outputs=out, name="Autoencoder")


# ──────────────────────────────────────────────
# Composite inference helper
# ──────────────────────────────────────────────

def inpaint_composite(
    autoencoder: Model,
    masked_image: tf.Tensor,
    mask: tf.Tensor,
) -> tf.Tensor:
    """
    Run autoencoder and paste prediction ONLY into the masked region.
    Known pixels are preserved exactly from the original input.

    Args:
        masked_image : (B, H, W, 3)  input with hole zeroed out
        mask         : (B, H, W, 1)  1=hole  0=known

    Returns:
        composite : (B, H, W, 3)
    """
    ae_input  = tf.concat([masked_image, mask], axis=-1)
    predicted = autoencoder(ae_input, training=False)
    composite = masked_image * (1.0 - mask) + predicted * mask
    return composite


# ──────────────────────────────────────────────
# Quick check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    dec = build_decoder()
    dec.summary()

    ae = build_autoencoder()
    ae.summary()

    dummy = tf.random.normal((2, 256, 256, 4))
    out   = ae(dummy, training=False)
    print("\nAutoencoder output shape:", out.shape)    # (2, 256, 256, 3)
    print("Output range: min={:.3f}  max={:.3f}".format(
        float(tf.reduce_min(out)), float(tf.reduce_max(out))
    ))
