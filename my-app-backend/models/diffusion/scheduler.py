"""
scheduler.py — DDPM (Denoising Diffusion Probabilistic Model) noise scheduler

The scheduler controls how noise is added (forward process) and removed
(reverse process) during training and inference.

Key idea for inpainting:
    - Forward  : add noise ONLY to the masked region
    - Reverse  : denoise step by step, conditioning on known pixels each step
    - Result   : known pixels stay intact, masked region is reconstructed
"""

import numpy as np
import tensorflow as tf


class DDPMScheduler:
    """
    Linear beta schedule DDPM as in Ho et al. 2020.

    Attributes:
        T          : total diffusion timesteps (default 1000)
        betas      : noise schedule β_1 ... β_T  shape (T,)
        alphas     : 1 - β                        shape (T,)
        alpha_bars : cumulative product of alphas  shape (T,)
    """

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T

        # Linear beta schedule
        self.betas      = np.linspace(beta_start, beta_end, T, dtype=np.float32)
        self.alphas     = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas).astype(np.float32)

        # Precompute useful terms
        self.sqrt_alpha_bars       = np.sqrt(self.alpha_bars)
        self.sqrt_one_minus_ab     = np.sqrt(1.0 - self.alpha_bars)
        self.sqrt_recip_alpha_bars = np.sqrt(1.0 / self.alpha_bars)
        self.sqrt_recip_m1_ab      = np.sqrt(1.0 / self.alpha_bars - 1.0)

        # Convert to tf constants for use inside @tf.function
        self._ab_tf     = tf.constant(self.alpha_bars)
        self._sqab_tf   = tf.constant(self.sqrt_alpha_bars)
        self._sq1mab_tf = tf.constant(self.sqrt_one_minus_ab)
        self._beta_tf   = tf.constant(self.betas)
        self._alpha_tf  = tf.constant(self.alphas)

    # ──────────────────────────────────────────
    # Forward process  q(x_t | x_0)
    # ──────────────────────────────────────────

    def add_noise(
        self,
        x0:        tf.Tensor,
        t:         tf.Tensor,
        noise:     tf.Tensor = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Sample x_t from q(x_t | x_0) = N(sqrt(ᾱ_t) x_0, (1-ᾱ_t) I)

        Args:
            x0    : clean image  (B, H, W, C)  in [-1, 1]
            t     : timestep     (B,)           int32
            noise : optional pre-sampled noise  (B, H, W, C)

        Returns:
            x_t   : noisy image  (B, H, W, C)
            noise : the noise added (needed for loss computation)
        """
        if noise is None:
            noise = tf.random.normal(tf.shape(x0))

        sqrt_ab   = tf.gather(self._sqab_tf,   t)           # (B,)
        sqrt_1mab = tf.gather(self._sq1mab_tf, t)           # (B,)

        # Reshape for broadcasting: (B,) → (B, 1, 1, 1)
        sqrt_ab   = tf.reshape(sqrt_ab,   [-1, 1, 1, 1])
        sqrt_1mab = tf.reshape(sqrt_1mab, [-1, 1, 1, 1])

        x_t = sqrt_ab * x0 + sqrt_1mab * noise
        return x_t, noise

    def add_noise_to_mask(
        self,
        x0:   tf.Tensor,
        mask: tf.Tensor,
        t:    tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Inpainting-specific forward process:
        Only add noise to the masked (unknown) region.
        Known pixels remain clean.

        Args:
            x0   : clean image   (B, H, W, 3)
            mask : binary mask   (B, H, W, 1)  1=hole
            t    : timesteps     (B,)

        Returns:
            x_t_masked : noisy in hole, clean outside  (B, H, W, 3)
            noise      : noise added to hole region    (B, H, W, 3)
        """
        noise = tf.random.normal(tf.shape(x0))
        x_t, noise = self.add_noise(x0, t, noise)

        # Blend: noisy in hole, clean outside
        x_t_masked = x0 * (1.0 - mask) + x_t * mask
        return x_t_masked, noise

    # ──────────────────────────────────────────
    # Reverse process  p(x_{t-1} | x_t)
    # ──────────────────────────────────────────

    def step(
        self,
        x_t:            tf.Tensor,
        t:              int,
        predicted_noise: tf.Tensor,
        x0_known:       tf.Tensor,
        mask:           tf.Tensor,
    ) -> tf.Tensor:
        """
        One reverse diffusion step: x_t → x_{t-1}

        After each step, paste the known (unmasked) pixels back from the
        original image — this is the RePaint-style inpainting trick that
        keeps known regions pixel-perfect.

        Args:
            x_t             : current noisy image    (B, H, W, 3)
            t               : current timestep       int
            predicted_noise : U-Net noise prediction (B, H, W, 3)
            x0_known        : original clean image   (B, H, W, 3)
            mask            : binary mask            (B, H, W, 1)

        Returns:
            x_{t-1} : less noisy image (B, H, W, 3)
        """
        beta_t    = self.betas[t]
        alpha_t   = self.alphas[t]
        alpha_bar = self.alpha_bars[t]

        # Posterior mean coefficient
        coef = beta_t / np.sqrt(1.0 - alpha_bar)

        # Mean of p(x_{t-1} | x_t)
        mean = (1.0 / np.sqrt(alpha_t)) * (x_t - coef * predicted_noise)

        if t == 0:
            x_prev = mean
        else:
            # Add posterior noise
            alpha_bar_prev = self.alpha_bars[t - 1]
            posterior_var  = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
            noise          = tf.random.normal(tf.shape(x_t))
            x_prev         = mean + np.sqrt(posterior_var) * noise

        # RePaint: re-inject known pixels at this noise level
        if t > 0:
            known_noisy, _ = self.add_noise(
                x0_known, tf.fill([tf.shape(x_t)[0]], t - 1)
            )
            x_prev = x0_known * (1.0 - mask) + x_prev * mask
        else:
            x_prev = x0_known * (1.0 - mask) + x_prev * mask

        return x_prev

    def sample_timesteps(self, batch_size: int) -> tf.Tensor:
        """Sample random timesteps for training — uniform over [0, T)."""
        return tf.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=self.T,
            dtype=tf.int32,
        )
