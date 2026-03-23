"""
unet.py — Conditional U-Net denoiser for DDPM inpainting

The U-Net takes:
    - x_t       : noisy image at timestep t  (B, H, W, 3)
    - mask      : binary inpainting mask     (B, H, W, 1)
    - t_embed   : sinusoidal timestep embed  (B, H, W, 1)  broadcast

And predicts the noise ε that was added to produce x_t.
The model learns p(ε | x_t, mask, t) so we can reverse the process.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model


# ──────────────────────────────────────────────
# Sinusoidal timestep embedding
# ──────────────────────────────────────────────

def sinusoidal_embedding(timesteps: tf.Tensor, dim: int = 256) -> tf.Tensor:
    """
    Sinusoidal position encoding for timesteps (from Attention is All You Need).
    Encodes scalar timestep t into a rich vector so the U-Net knows
    how much noise is present at each step.

    Args:
        timesteps : (B,) int32 timestep values
        dim       : embedding dimension

    Returns:
        embedding : (B, dim) float32
    """
    half  = dim // 2
    freqs = tf.exp(
        -np.log(10000.0) * tf.cast(tf.range(half), tf.float32) / (half - 1)
    )                                                          # (half,)

    args  = tf.cast(timesteps[:, None], tf.float32) * freqs[None]  # (B, half)
    embed = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)       # (B, dim)
    return embed


def timestep_embedding_layer(t: tf.Tensor, channels: int) -> tf.Tensor:
    """Project sinusoidal embedding to match spatial feature channels."""
    emb = sinusoidal_embedding(t, dim=256)                    # (B, 256)
    emb = layers.Dense(channels, activation="silu")(emb)      # (B, channels)
    emb = layers.Dense(channels)(emb)                          # (B, channels)
    return emb


# ──────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────

def resnet_block(x, t_emb, filters: int):
    """
    ResNet block with timestep conditioning.
    t_emb is added to the feature map so the model adapts per noise level.
    Uses LayerNormalization (compatible with TF 2.13 / Keras 2).
    """
    residual = x

    x = layers.LayerNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)

    # Add timestep embedding (broadcast over spatial dims)
    t_proj = layers.Dense(filters)(layers.Activation("swish")(t_emb))
    t_proj = t_proj[:, None, None, :]                          # (B,1,1,C)
    x = x + t_proj

    x = layers.LayerNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)

    # Residual projection if channel mismatch
    if residual.shape[-1] != filters:
        residual = layers.Conv2D(filters, 1)(residual)

    return x + residual


def downsample(x, filters: int):
    """Strided conv downsampling — halves spatial dims."""
    return layers.Conv2D(filters, 4, strides=2, padding="same")(x)


def upsample(x, filters: int):
    """Transposed conv upsampling — doubles spatial dims."""
    return layers.Conv2DTranspose(filters, 4, strides=2, padding="same")(x)


def attention_block(x):
    """
    Self-attention block at bottleneck.
    Lets the model reason about long-range dependencies —
    critical for coherent inpainting of large holes.
    """
    B, H, W, C = tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]

    x_norm = layers.LayerNormalization()(x)
    flat   = tf.reshape(x_norm, [B, H * W, C])                # (B, N, C)

    q = layers.Dense(C)(flat)
    k = layers.Dense(C)(flat)
    v = layers.Dense(C)(flat)

    scale   = tf.cast(C, tf.float32) ** -0.5
    attn    = tf.matmul(q, k, transpose_b=True) * scale       # (B, N, N)
    attn    = tf.nn.softmax(attn, axis=-1)
    out     = tf.matmul(attn, v)                               # (B, N, C)

    out     = tf.reshape(out, [B, H, W, C])
    out     = layers.Conv2D(C, 1)(out)
    return x + out


# ──────────────────────────────────────────────
# Conditional U-Net
# ──────────────────────────────────────────────

def build_unet(image_size: int = 256, base_channels: int = 64) -> Model:
    """
    Conditional U-Net denoiser for DDPM inpainting.

    Inputs:
        noisy_image : (B, H, W, 3)  x_t — noisy image at timestep t
        mask        : (B, H, W, 1)  binary inpainting mask
        timestep    : (B,)          int32 current diffusion timestep

    Output:
        noise_pred  : (B, H, W, 3)  predicted noise ε

    Architecture:
        Input (5ch) → Encoder → Bottleneck (+ attention) → Decoder → Output
        Timestep embedding injected at every ResNet block
        Skip connections from encoder to decoder (U-Net style)
    """
    ch = base_channels   # 64

    # ── Inputs ───────────────────────────────
    noisy_image = layers.Input(shape=(image_size, image_size, 3), name="noisy_image")
    mask        = layers.Input(shape=(image_size, image_size, 1), name="mask")
    timestep    = layers.Input(shape=(),                           name="timestep", dtype=tf.int32)

    # Concatenate inputs: noisy image + mask = 4 channels
    x = layers.Concatenate()([noisy_image, mask])              # (B,256,256,4)
    x = layers.Conv2D(ch, 3, padding="same")(x)               # (B,256,256,64)

    # ── Timestep embedding ────────────────────
    t_emb = timestep_embedding_layer(timestep, ch * 4)         # (B, 256)

    # ── Encoder ──────────────────────────────
    e1 = resnet_block(x,  t_emb, ch)                           # (256,256,64)
    e2 = resnet_block(downsample(e1, ch*2), t_emb, ch*2)       # (128,128,128)
    e3 = resnet_block(downsample(e2, ch*4), t_emb, ch*4)       # (64, 64, 256)
    e4 = resnet_block(downsample(e3, ch*8), t_emb, ch*8)       # (32, 32, 512)

    # ── Bottleneck + self-attention ───────────
    b  = resnet_block(downsample(e4, ch*8), t_emb, ch*8)       # (16, 16, 512)
    b  = attention_block(b)
    b  = resnet_block(b, t_emb, ch*8)                          # (16, 16, 512)

    # ── Decoder with skip connections ─────────
    d4 = resnet_block(
        layers.Concatenate()([upsample(b,  ch*8), e4]), t_emb, ch*8)  # (32,32,512)
    d3 = resnet_block(
        layers.Concatenate()([upsample(d4, ch*4), e3]), t_emb, ch*4)  # (64,64,256)
    d2 = resnet_block(
        layers.Concatenate()([upsample(d3, ch*2), e2]), t_emb, ch*2)  # (128,128,128)
    d1 = resnet_block(
        layers.Concatenate()([upsample(d2, ch),   e1]), t_emb, ch)    # (256,256,64)

    # ── Output head ──────────────────────────
    out = layers.LayerNormalization()(d1)
    out = layers.Activation("swish")(out)
    out = layers.Conv2D(3, 3, padding="same", name="noise_pred")(out)  # (256,256,3)

    return Model(
        inputs=[noisy_image, mask, timestep],
        outputs=out,
        name="Conditional_UNet_Denoiser",
    )


# ──────────────────────────────────────────────
# Quick check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    unet = build_unet()
    unet.summary()

    b         = 2
    img       = tf.random.normal((b, 256, 256, 3))
    mask      = tf.random.uniform((b, 256, 256, 1))
    timesteps = tf.random.uniform((b,), 0, 1000, dtype=tf.int32)

    out = unet([img, mask, timesteps], training=False)
    print("\nNoise prediction shape:", out.shape)  # (2, 256, 256, 3)
