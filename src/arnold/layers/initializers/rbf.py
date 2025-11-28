# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Radial basis function parameter initializer.

Provides initialization strategies optimized for RBF center and width parameters.
"""

from __future__ import annotations

import tensorflow as tf

from arnold.layers.initializers.base import KANInitializer


tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold")
class RBFInitializer(KANInitializer):
    r"""
    Initializer for radial basis function parameters.

    Provides smart initialization for:

    - **Centers**: Uniformly spaced or random placement.
    - **Widths**: Based on center spacing for overlap control.

    For uniform center spacing in :math:`[a, b]` with :math:`n` centers:

    .. math::

        c_k = a + (b - a) \cdot \frac{k}{n-1}, \quad k = 0, \ldots, n-1

    Default width ensures neighboring RBFs overlap:

    .. math::

        \sigma = \frac{b - a}{n - 1} \cdot \text{overlap\_factor}

    Parameters
    ----------
    num_centers : int
        Number of RBF centers.
    domain : tuple[float, float], default=(-1, 1)
        Domain interval for center placement.
    param_type : {'centers', 'widths'}, default='centers'
        Which parameter type to initialize.
    mode : {'uniform', 'random'}, default='uniform'
        Placement strategy.
    overlap_factor : float, default=1.0
        Width scaling relative to center spacing (for 'widths' param_type).
    seed : int | None, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> # Initialize uniformly spaced centers
    >>> center_init = RBFInitializer(num_centers=20, param_type='centers')
    >>>
    >>> # Initialize widths with 50% overlap
    >>> width_init = RBFInitializer(num_centers=20, param_type='widths',
    ...                              overlap_factor=1.5)
    """

    def __init__(
        self,
        num_centers: int,
        domain: tuple[float, float] = (-1.0, 1.0),
        param_type: str = "centers",
        mode: str = "uniform",
        overlap_factor: float = 1.0,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if param_type not in ("centers", "widths"):
            raise ValueError(
                f"param_type must be 'centers' or 'widths', got {param_type}"
            )
        if mode not in ("uniform", "random"):
            raise ValueError(f"mode must be 'uniform' or 'random', got {mode}")
        self.num_centers = int(num_centers)
        self.domain = (float(domain[0]), float(domain[1]))
        self.param_type = param_type
        self.mode = mode
        self.overlap_factor = float(overlap_factor)
        self.seed = seed

    def __call__(self, shape, dtype=None) -> tf.Tensor:
        """Generate RBF parameter initialization."""
        dtype = dtype or tf.float32
        a, b = self.domain

        if self.param_type == "centers":
            return self._init_centers(shape, dtype, a, b)
        else:
            return self._init_widths(shape, dtype, a, b)

    def _init_centers(self, shape, dtype, a: float, b: float) -> tf.Tensor:
        """Initialize RBF centers."""
        if self.mode == "uniform":
            # Uniformly spaced centers
            n = shape[-1] if len(shape) >= 1 else self.num_centers
            centers_1d = tf.linspace(
                tf.cast(a, dtype), tf.cast(b, dtype), n
            )
            # Broadcast to full shape
            result = tf.broadcast_to(centers_1d, shape)
            return tf.cast(result, dtype)
        else:
            # Random placement
            generator = (
                tf.random.Generator.from_seed(self.seed) if self.seed else None
            )
            if generator:
                return generator.uniform(shape, a, b, dtype=dtype)
            else:
                return tf.random.uniform(shape, a, b, dtype=dtype)

    def _init_widths(self, shape, dtype, a: float, b: float) -> tf.Tensor:
        """Initialize RBF widths based on center spacing."""
        n = shape[-1] if len(shape) >= 1 else self.num_centers
        # Default width: spacing between centers * overlap factor
        spacing = (b - a) / max(n - 1, 1)
        width = spacing * self.overlap_factor

        return tf.fill(shape, tf.cast(width, dtype))

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "num_centers": self.num_centers,
                "domain": self.domain,
                "param_type": self.param_type,
                "mode": self.mode,
                "overlap_factor": self.overlap_factor,
                "seed": self.seed,
            }
        )
        return config
