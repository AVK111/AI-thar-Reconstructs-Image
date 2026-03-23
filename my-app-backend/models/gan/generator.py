import tensorflow as tf
from tensorflow.keras import layers, Model


# ──────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────

def conv_block(x, filters, kernel_size=4, strides=2, use_bn=True, activation="leaky_relu"):
    """Encoder conv block: Conv → BatchNorm → Activation."""
    x = layers.Conv2D(
        filters, kernel_size,
        strides=strides,
        padding="same",
        use_bias=not use_bn,
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    if activation == "leaky_relu":
        x = layers.LeakyReLU(0.2)(x)
    elif activation == "relu":
        x = layers.ReLU()(x)
    return x


def deconv_block(x, skip, filters, kernel_size=4, dropout=False):
    """
    Decoder block: ConvTranspose → BatchNorm → (Dropout) → ReLU → skip concat.
    Skip connection concatenated AFTER upsampling (U-Net style).
    """
    x = layers.Conv2DTranspose(
        filters, kernel_size,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(x)
    x = layers.BatchNormalization()(x)
    if dropout:
        x = layers.Dropout(0.5)(x)
    x = layers.ReLU()(x)
    if skip is not None:
        x = layers.Concatenate()([x, skip])
    return x


# ──────────────────────────────────────────────
# Gated convolution block
# ──────────────────────────────────────────────

def gated_conv(x, filters, kernel_size=3, strides=1, dilation_rate=1, use_bn=True):
    """
    Gated convolution: splits channels into feature + gate, applies sigmoid gate.
    Helps the model distinguish known vs unknown regions naturally.
    """
    feature = layers.Conv2D(
        filters, kernel_size,
        strides=strides,
        padding="same",
        dilation_rate=dilation_rate,
        use_bias=not use_bn,
    )(x)
    gate = layers.Conv2D(
        filters, kernel_size,
        strides=strides,
        padding="same",
        dilation_rate=dilation_rate,
        use_bias=not use_bn,
    )(x)
    if use_bn:
        feature = layers.BatchNormalization()(feature)
        gate    = layers.BatchNormalization()(gate)
    feature = layers.LeakyReLU(0.2)(feature)
    gate    = layers.Activation("sigmoid")(gate)
    return feature * gate


# ──────────────────────────────────────────────
# U-Net Generator
# ──────────────────────────────────────────────

def build_generator(image_size: int = 256) -> Model:
    """
    U-Net Generator for image inpainting.

    Input  : masked RGB image (3ch) concatenated with binary mask (1ch) → 4ch
    Output : inpainted RGB image in [-1, 1]

    Architecture
    ============
    Encoder     : 256→128→64→32→16→8→4
    Bottleneck  : dilated gated convolutions at 4×4 (rates 1,2,4,8,4,2,1)
    Decoder     : 4→8→16→32→64→128→256  with U-Net skip connections

    Skip connections:
        d1 (8×8)   ← e5 (8×8)
        d2 (16×16) ← e4 (16×16)
        d3 (32×32) ← e3 (32×32)
        d4 (64×64) ← e2 (64×64)
        d5 (128×128) ← e1 (128×128)
        d6 (256×256) ← no skip (full resolution)
    """
    inp = layers.Input(shape=(image_size, image_size, 4), name="masked_image_and_mask")

    # Encoder — reduced filters to fit in 1.6GB VRAM (RTX 3050 Laptop)
    e1 = conv_block(inp, 32,  use_bn=False)     # (128,128, 32)
    e2 = conv_block(e1,  64)                    # (64, 64,  64)
    e3 = conv_block(e2,  128)                   # (32, 32, 128)
    e4 = conv_block(e3,  256)                   # (16, 16, 256)
    e5 = conv_block(e4,  256)                   # (8,  8,  256)
    e6 = conv_block(e5,  256)                   # (4,  4,  256) bottleneck input

    # Bottleneck: dilated gated convolutions stays at (4, 4, 256)
    b = gated_conv(e6, 256, dilation_rate=1)
    b = gated_conv(b,  256, dilation_rate=2)
    b = gated_conv(b,  256, dilation_rate=4)
    b = gated_conv(b,  256, dilation_rate=8)
    b = gated_conv(b,  256, dilation_rate=4)
    b = gated_conv(b,  256, dilation_rate=2)
    b = gated_conv(b,  256, dilation_rate=1)    # (4, 4, 256)

    # Decoder with skip connections from matching encoder levels
    d1 = deconv_block(b,  e5, 256, dropout=True)   # (8,  8,  512)
    d2 = deconv_block(d1, e4, 256, dropout=True)   # (16, 16, 512)
    d3 = deconv_block(d2, e3, 128, dropout=True)   # (32, 32, 256)
    d4 = deconv_block(d3, e2, 64)                  # (64, 64, 128)
    d5 = deconv_block(d4, e1, 32)                  # (128,128,  64)
    d6 = deconv_block(d5, None, 32)                # (256,256,  32)

    # Output head
    out = layers.Conv2DTranspose(
        3, kernel_size=4,
        strides=1,
        padding="same",
        activation="tanh",
        name="inpainted_output",
        dtype="float32",   # always output float32 even with mixed precision
    )(d6)                                           # (256,256,3) in [-1,1]

    return Model(inputs=inp, outputs=out, name="UNet_Generator")


# ──────────────────────────────────────────────
# Composite forward pass helper
# ──────────────────────────────────────────────

def inpaint_composite(generator: Model, masked_image: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """
    Run generator and paste prediction ONLY into the masked region.
    Known pixels are preserved exactly from the original input.

    Args:
        masked_image : (B, H, W, 3)  input with masked region zeroed out
        mask         : (B, H, W, 1)  1=missing  0=known

    Returns:
        composite : (B, H, W, 3)
    """
    gen_input  = tf.concat([masked_image, mask], axis=-1)
    prediction = generator(gen_input, training=False)
    composite  = masked_image * (1.0 - mask) + prediction * mask
    return composite


# ──────────────────────────────────────────────
# Quick check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    gen = build_generator()
    gen.summary()
    dummy = tf.random.normal((2, 256, 256, 4))
    out   = gen(dummy, training=False)
    print("\nOutput shape:", out.shape)
    print("Output range: min={:.3f}  max={:.3f}".format(
        float(tf.reduce_min(out)), float(tf.reduce_max(out))
    ))
