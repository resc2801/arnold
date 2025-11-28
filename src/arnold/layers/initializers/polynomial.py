# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Polynomial coefficient initializer.

Provides initialization strategies optimized for polynomial basis expansions.
"""

from __future__ import annotations

import tensorflow as tf

from arnold.layers.initializers.base import KANInitializer


tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold")
class PolynomialInitializer(KANInitializer):
    r"""
    Initializer for polynomial basis coefficients.

    Supports multiple initialization strategies optimized for polynomial
    expansions:

    - **uniform**: Uniform random in :math:`[-a, a]` with degree scaling.
    - **normal**: Normal distribution with degree-dependent variance.
    - **zeros**: Zero initialization (for fine-tuning).
    - **identity**: Identity-like initialization for linear behavior.

    The degree scaling factor accounts for the growth of polynomial magnitudes:

    .. math::

        \sigma_k = \frac{\sigma_0}{\sqrt{k+1}}

    where :math:`k` is the polynomial degree.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.
    mode : {'uniform', 'normal', 'zeros', 'identity'}, default='normal'
        Initialization mode.
    scale : float, default=0.05
        Base scale factor.
    normalize : bool, default=True
        Whether to apply degree-dependent scaling.
    seed : int | None, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> init = PolynomialInitializer(degree=10, mode='normal', scale=0.05)
    >>> coeffs = init(shape=(32, 10, 11))  # (in_features, out_units, degree+1)
    """

    def __init__(
        self,
        degree: int,
        mode: str = "normal",
        scale: float = 0.05,
        normalize: bool = True,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if mode not in ("uniform", "normal", "zeros", "identity"):
            raise ValueError(
                f"mode must be 'uniform', 'normal', 'zeros', or 'identity', "
                f"got {mode}"
            )
        self.degree = int(degree)
        self.mode = mode
        self.scale = float(scale)
        self.normalize = normalize
        self.seed = seed

    def __call__(self, shape, dtype=None) -> tf.Tensor:
        """Generate polynomial coefficient initialization."""
        dtype = dtype or tf.float32

        if self.mode == "zeros":
            return tf.zeros(shape, dtype=dtype)

        if self.mode == "identity":
            # Initialize to approximate identity (linear term only)
            result = tf.zeros(shape, dtype=dtype)
            if len(shape) >= 1 and shape[-1] > 1:
                # Set coefficient of degree-1 term
                indices = tf.concat(
                    [
                        tf.zeros([tf.reduce_prod(shape[:-1]), len(shape) - 1], tf.int32),
                        tf.ones([tf.reduce_prod(shape[:-1]), 1], tf.int32),
                    ],
                    axis=-1,
                )
                updates = tf.ones([tf.reduce_prod(shape[:-1])], dtype=dtype)
                result = tf.tensor_scatter_nd_update(result, indices, updates)
            return result

        # Generate random initialization
        generator = tf.random.Generator.from_seed(self.seed) if self.seed else None

        if self.mode == "uniform":
            if generator:
                base = generator.uniform(shape, -1.0, 1.0, dtype=dtype)
            else:
                base = tf.random.uniform(shape, -1.0, 1.0, dtype=dtype)
        else:  # normal
            if generator:
                base = generator.normal(shape, 0.0, 1.0, dtype=dtype)
            else:
                base = tf.random.normal(shape, 0.0, 1.0, dtype=dtype)

        # Apply scaling
        result = self.scale * base

        # Apply degree-dependent normalization if requested
        if self.normalize and len(shape) >= 1:
            # Create scaling factors 1/sqrt(k+1) for k = 0, 1, ..., degree
            num_terms = shape[-1]
            degree_scale = tf.cast(
                1.0 / tf.sqrt(tf.range(1, num_terms + 1, dtype=tf.float32)),
                dtype=dtype,
            )
            result = result * degree_scale

        return result

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "degree": self.degree,
                "mode": self.mode,
                "scale": self.scale,
                "normalize": self.normalize,
                "seed": self.seed,
            }
        )
        return config
