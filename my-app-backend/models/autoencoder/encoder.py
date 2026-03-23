import tensorflow as tf
from tensorflow.keras import layers, Model


# ──────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────

def build_encoder(image_size: int = 256, latent_dim: int = 512) -> Model:
    """
    Convolutional Encoder for image inpainting.

    Takes a 4-channel input (masked RGB + binary mask) and compresses
    it into a dense latent vector capturing semantic content of the image.

    Input  : (B, 256, 256, 4)  masked image + mask
    Output : (B, latent_dim)   latent representation

    Architecture: Progressive downsampling with residual connections
    256 → 128 → 64 → 32 → 16 → 8 → flatten → dense latent
    """
    inp = layers.Input(shape=(image_size, image_size, 4), name="encoder_input")

    # ── Downsampling blocks ───────────────────
    x = layers.Conv2D(64, 4, strides=2, padding="same", use_bias=False)(inp)
    x = layers.LeakyReLU(0.2)(x)                              # (128,128,64)

    x = layers.Conv2D(128, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)                              # (64,64,128)

    x = layers.Conv2D(256, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)                              # (32,32,256)

    x = layers.Conv2D(512, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)                              # (16,16,512)

    x = layers.Conv2D(512, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)                              # (8,8,512)

    x = layers.Conv2D(512, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)                              # (4,4,512)

    # ── Bottleneck → latent vector ────────────
    x = layers.Flatten()(x)                                   # (8192,)
    x = layers.Dense(latent_dim * 2, activation="relu")(x)
    latent = layers.Dense(latent_dim, name="latent_vector")(x) # (latent_dim,)

    return Model(inputs=inp, outputs=latent, name="Encoder")


# ──────────────────────────────────────────────
# Quick check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    enc = build_encoder()
    enc.summary()
    dummy = tf.random.normal((2, 256, 256, 4))
    out   = enc(dummy, training=False)
    print("\nLatent shape:", out.shape)   # (2, 512)
