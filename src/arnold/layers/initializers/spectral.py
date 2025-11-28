# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Spectral/Fourier coefficient initializer.

Provides initialization strategies optimized for spectral basis expansions.
"""

from __future__ import annotations

import tensorflow as tf

from arnold.layers.initializers.base import KANInitializer

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold")
class SpectralInitializer(KANInitializer):
    r"""
    Initializer for spectral/Fourier basis coefficients.

    Supports initialization strategies that account for spectral decay:

    - **power_law**: Coefficients decay as :math:`1/k^{\alpha}`.
    - **exponential**: Coefficients decay as :math:`e^{-\alpha k}`.
    - **uniform**: Uniform random (no decay).
    - **zeros**: Zero initialization.

    Power-law initialization models smooth functions:

    .. math::

        c_k \sim \frac{1}{(k+1)^\alpha}

    where :math:`\alpha > 0` controls the smoothness (higher = smoother).

    Parameters
    ----------
    num_frequencies : int
        Number of frequency components.
    mode : {'power_law', 'exponential', 'uniform', 'zeros'}, default='power_law'
        Decay mode for coefficient magnitudes.
    decay_rate : float, default=1.0
        Decay rate parameter (:math:`\alpha`).
    scale : float, default=1.0
        Overall magnitude scaling.
    seed : int | None, optional
        Random seed for reproducibility.

    Notes
    -----
    For Fourier bases, coefficients typically come in (cos, sin) pairs.
    This initializer handles both cases based on shape.

    Examples
    --------
    >>> # Power-law decay for smooth target functions
    >>> init = SpectralInitializer(num_frequencies=50, mode='power_law',
    ...                             decay_rate=2.0)
    >>>
    >>> # Exponential decay for analytic functions
    >>> init = SpectralInitializer(num_frequencies=50, mode='exponential',
    ...                             decay_rate=0.1)
    """

    def __init__(
        self,
        num_frequencies: int,
        mode: str = "power_law",
        decay_rate: float = 1.0,
        scale: float = 1.0,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if mode not in ("power_law", "exponential", "uniform", "zeros"):
            raise ValueError(
                f"mode must be 'power_law', 'exponential', 'uniform', or "
                f"'zeros', got {mode}"
            )
        self.num_frequencies = int(num_frequencies)
        self.mode = mode
        self.decay_rate = float(decay_rate)
        self.scale = float(scale)
        self.seed = seed

    def __call__(self, shape, dtype=None) -> tf.Tensor:
        """Generate spectral coefficient initialization."""
        dtype = dtype or tf.float32

        if self.mode == "zeros":
            return tf.zeros(shape, dtype=dtype)

        # Generate base random values
        generator = tf.random.Generator.from_seed(self.seed) if self.seed else None
        if generator:
            base = generator.normal(shape, 0.0, 1.0, dtype=dtype)
        else:
            base = tf.random.normal(shape, 0.0, 1.0, dtype=dtype)

        if self.mode == "uniform":
            return self.scale * base

        # Apply spectral decay
        num_terms = shape[-1] if len(shape) >= 1 else self.num_frequencies
        k = tf.range(1, num_terms + 1, dtype=dtype)

        if self.mode == "power_law":
            decay = tf.pow(k, -self.decay_rate)
        else:  # exponential
            decay = tf.exp(-self.decay_rate * k)

        # Apply decay and scaling
        result = self.scale * base * decay

        return result

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "num_frequencies": self.num_frequencies,
                "mode": self.mode,
                "decay_rate": self.decay_rate,
                "scale": self.scale,
                "seed": self.seed,
            }
        )
        return config
