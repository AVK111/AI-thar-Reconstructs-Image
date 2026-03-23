import tensorflow as tf


# ──────────────────────────────────────────────
# Loss weights  (tune these during experiments)
# ──────────────────────────────────────────────
LAMBDA_L1        = 10.0    # pixel-wise reconstruction (whole image)
LAMBDA_L1_HOLE   = 30.0    # extra weight on the inpainted region
LAMBDA_PERCEPTUAL = 0.1    # VGG perceptual loss
LAMBDA_STYLE     = 50.0    # Gram matrix style loss
LAMBDA_GAN       = 1.0     # adversarial loss weight


# ──────────────────────────────────────────────
# Adversarial losses  (non-saturating + LSGAN)
# ──────────────────────────────────────────────

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
    """
    Standard non-saturating GAN discriminator loss.
    real_output / fake_output : raw logit patch maps from PatchGAN
    """
    real_loss = bce(tf.ones_like(real_output),  real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    return (real_loss + fake_loss) * 0.5


def generator_adversarial_loss(fake_output: tf.Tensor) -> tf.Tensor:
    """Generator tries to fool discriminator — all patches labelled real."""
    return bce(tf.ones_like(fake_output), fake_output)


# ──────────────────────────────────────────────
# Reconstruction losses
# ──────────────────────────────────────────────

def reconstruction_loss(
    real: tf.Tensor,
    fake: tf.Tensor,
    mask: tf.Tensor,
) -> tf.Tensor:
    """
    Weighted L1 loss:
        - masked (hole) region gets LAMBDA_L1_HOLE weight
        - valid (known) region gets LAMBDA_L1 weight

    Args:
        real : ground-truth image   (B, H, W, 3)
        fake : generated image      (B, H, W, 3)
        mask : binary mask          (B, H, W, 1)  1=hole
    """
    real = tf.cast(real, tf.float32)
    fake = tf.cast(fake, tf.float32)
    mask = tf.cast(mask, tf.float32)
    diff      = tf.abs(real - fake)
    hole_loss = tf.reduce_mean(diff * mask)         * LAMBDA_L1_HOLE
    valid_loss= tf.reduce_mean(diff * (1.0 - mask)) * LAMBDA_L1
    return tf.cast(hole_loss + valid_loss, tf.float32)


# ──────────────────────────────────────────────
# Perceptual + Style losses  (VGG16 features)
# ──────────────────────────────────────────────

def _build_vgg_feature_extractor():
    """Return VGG16 sub-model outputting relu1_2, relu2_2, relu3_3 activations."""
    vgg = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
    vgg.trainable = False
    layer_names = ["block1_conv2", "block2_conv2", "block3_conv3"]
    outputs = [vgg.get_layer(n).output for n in layer_names]
    return tf.keras.Model(inputs=vgg.input, outputs=outputs, name="VGG_features")


_vgg_extractor = None   # lazy-initialised to avoid loading weights at import time


def _get_vgg():
    global _vgg_extractor
    if _vgg_extractor is None:
        _vgg_extractor = _build_vgg_feature_extractor()
    return _vgg_extractor


def _vgg_preprocess(x: tf.Tensor) -> tf.Tensor:
    """[-1,1] → VGG BGR mean-subtracted."""
    x = tf.cast(x, tf.float32)              # ensure float32 for VGG
    x = (x + 1.0) * 127.5                  # [-1,1] → [0,255]
    return tf.keras.applications.vgg16.preprocess_input(x)


def perceptual_loss(real: tf.Tensor, fake: tf.Tensor) -> tf.Tensor:
    """L1 distance between VGG feature maps of real and generated images."""
    vgg    = _get_vgg()
    r_feat = vgg(_vgg_preprocess(real), training=False)
    f_feat = vgg(_vgg_preprocess(fake), training=False)
    loss   = sum(tf.reduce_mean(tf.abs(r - f)) for r, f in zip(r_feat, f_feat))
    return loss * LAMBDA_PERCEPTUAL


def _gram_matrix(feat: tf.Tensor) -> tf.Tensor:
    """Gram matrix of a feature map (B, H, W, C) → (B, C, C)."""
    feat       = tf.cast(feat, tf.float32)   # ensure float32 for mixed precision
    b, h, w, c = tf.unstack(tf.shape(feat))
    feat_flat  = tf.reshape(feat, [b, h * w, c])
    gram       = tf.matmul(feat_flat, feat_flat, transpose_a=True)
    return gram / tf.cast(h * w * c, tf.float32)


def style_loss(real: tf.Tensor, fake: tf.Tensor) -> tf.Tensor:
    """Gram matrix style loss across VGG relu layers."""
    vgg    = _get_vgg()
    r_feat = vgg(_vgg_preprocess(real), training=False)
    f_feat = vgg(_vgg_preprocess(fake), training=False)
    loss   = sum(
        tf.reduce_mean(tf.abs(_gram_matrix(r) - _gram_matrix(f)))
        for r, f in zip(r_feat, f_feat)
    )
    return loss * LAMBDA_STYLE


# ──────────────────────────────────────────────
# Combined generator loss
# ──────────────────────────────────────────────

def generator_total_loss(
    real:             tf.Tensor,
    fake:             tf.Tensor,
    mask:             tf.Tensor,
    fake_global_out:  tf.Tensor,
    fake_local_out:   tf.Tensor,
) -> dict:
    """
    Aggregate all generator losses into a single scalar + per-component dict.

    Args:
        real            : ground-truth image        (B,H,W,3)
        fake            : generated/inpainted image (B,H,W,3)
        mask            : binary mask               (B,H,W,1)
        fake_global_out : global disc patch logits  (B,16,16,1)
        fake_local_out  : local  disc patch logits  (B,8,8,1)

    Returns:
        dict with keys: total, adv_global, adv_local, l1, perceptual, style
    """
    adv_g = generator_adversarial_loss(fake_global_out) * LAMBDA_GAN
    adv_l = generator_adversarial_loss(fake_local_out)  * LAMBDA_GAN
    l1    = reconstruction_loss(real, fake, mask)
    perc  = perceptual_loss(real, fake)
    sty   = style_loss(real, fake)

    total = adv_g + adv_l + l1 + perc + sty

    return {
        "total":      total,
        "adv_global": adv_g,
        "adv_local":  adv_l,
        "l1":         l1,
        "perceptual": perc,
        "style":      sty,
    }
