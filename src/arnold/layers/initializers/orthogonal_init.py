# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Orthogonal matrix initializer for KAN layers.

Provides orthonormal initialization for weight matrices.
"""

from __future__ import annotations

import tensorflow as tf

from arnold.layers.initializers.base import KANInitializer

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold")
class OrthogonalInitializer(KANInitializer):
    r"""
    Orthogonal matrix initializer for KAN weight matrices.

    Generates matrices with orthonormal rows or columns using QR decomposition
    of a random Gaussian matrix.

    For a random matrix :math:`A \sim \mathcal{N}(0, 1)`:

    .. math::

        A = QR \quad \Rightarrow \quad W = \text{gain} \cdot Q

    where :math:`Q` is orthogonal.

    Parameters
    ----------
    gain : float, default=1.0
        Multiplicative factor applied to the orthogonal matrix.
    seed : int | None, optional
        Random seed for reproducibility.

    Notes
    -----
    This is similar to :class:`tf.keras.initializers.Orthogonal` but
    integrated with the KANInitializer interface.

    Orthogonal initialization helps with:

    - Gradient flow in deep networks
    - Avoiding vanishing/exploding gradients
    - Maintaining signal magnitude through layers

    Examples
    --------
    >>> init = OrthogonalInitializer(gain=1.0)
    >>> W = init(shape=(64, 64))  # Orthogonal 64×64 matrix
    >>>
    >>> # With scaling for ReLU
    >>> init = OrthogonalInitializer(gain=tf.sqrt(2.0))
    """

    def __init__(self, gain: float = 1.0, seed: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.gain = float(gain)
        self.seed = seed

    def __call__(self, shape, dtype=None) -> tf.Tensor:
        """Generate orthogonal weight matrix."""
        dtype = dtype or tf.float32

        if len(shape) < 2:
            raise ValueError(
                f"OrthogonalInitializer requires at least 2D shape, got {shape}"
            )

        # Flatten to 2D for QR decomposition
        num_rows = shape[0]
        num_cols = tf.reduce_prod(shape[1:])
        flat_shape = (num_rows, num_cols)

        # Generate random Gaussian matrix
        generator = tf.random.Generator.from_seed(self.seed) if self.seed else None
        if generator:
            a = generator.normal(flat_shape, dtype=dtype)
        else:
            a = tf.random.normal(flat_shape, dtype=dtype)

        # QR decomposition
        if num_rows < num_cols:
            # More columns than rows: use transpose
            q, r = tf.linalg.qr(tf.transpose(a))
            q = tf.transpose(q)
        else:
            q, r = tf.linalg.qr(a)

        # Make Q deterministic (fix sign ambiguity)
        d = tf.linalg.diag_part(r)
        q = q * tf.sign(d)

        # Apply gain
        q = self.gain * q

        # Truncate or pad to match requested shape
        q = q[:num_rows, :num_cols]

        # Reshape to original shape
        return tf.reshape(q, shape)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"gain": self.gain, "seed": self.seed})
        return config
