# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Elliptic function KAN layer.

This module provides :class:`EllipticFunctions`, a KAN layer using Jacobi
elliptic functions as basis functions.

Mathematical Background
-----------------------
Jacobi elliptic functions are doubly periodic functions that generalize
trigonometric functions. The three main functions are:

- :math:`\text{sn}(u, k)`: Generalized sine
- :math:`\text{cn}(u, k)`: Generalized cosine
- :math:`\text{dn}(u, k)`: Delta amplitude

These satisfy:

.. math::

    \text{sn}^2(u, k) + \text{cn}^2(u, k) = 1
    k^2 \text{sn}^2(u, k) + \text{dn}^2(u, k) = 1

The parameter k (elliptic modulus) controls the shape:
- k → 0: sn → sin, cn → cos, dn → 1
- k → 1: sn → tanh, cn → sech, dn → sech

Applications:
- Pendulum motion
- Nonlinear wave equations (solitons)
- Conformal mapping
- Cryptography

References
----------
.. [1] NIST DLMF, Ch. 22: Jacobian Elliptic Functions
.. [2] Abramowitz & Stegun, Ch. 16: Jacobian Elliptic Functions
"""
import tensorflow as tf

from arnold.layers.core.special.base import SpecialBase


tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold", name="EllipticFunctions")
class EllipticFunctions(SpecialBase):
    r"""
    Kolmogorov-Arnold Network layer using Jacobi elliptic functions.

    Uses sn(u, k), cn(u, k), dn(u, k) at different frequencies as basis.

    Parameters
    ----------
    units : int
        Output dimension.
    max_order : int, default=8
        Number of frequency harmonics.
    modulus : float, default=0.5
        Elliptic modulus k ∈ [0, 1).
    learnable_modulus : bool, default=False
        Whether to make k a trainable parameter.
    input_clip : tuple[float, float] | None, default=None
        Input clipping for numerical stability.
    **kwargs
        Additional Keras layer arguments.

    Notes
    -----
    For k=0, elliptic functions reduce to trigonometric:
    - sn(u, 0) = sin(u)
    - cn(u, 0) = cos(u)
    - dn(u, 0) = 1

    For k→1, they become hyperbolic:
    - sn(u, 1) = tanh(u)
    - cn(u, 1) = sech(u)
    - dn(u, 1) = sech(u)

    Example
    -------
    >>> import tensorflow as tf
    >>> from arnold.layers.core.special import EllipticFunctions
    >>>
    >>> layer = EllipticFunctions(max_order=4, modulus=0.7, units=32)
    >>> x = tf.random.uniform((16, 10), -5, 5)
    >>> y = layer(x)  # shape: (16, 32)
    """

    def __init__(
        self,
        *,
        units: int,
        max_order: int = 8,
        modulus: float = 0.5,
        learnable_modulus: bool = False,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        if not 0 <= modulus < 1:
            raise ValueError(f"modulus must be in [0, 1), got {modulus}")
        self.initial_modulus = modulus
        self.learnable_modulus = learnable_modulus
        super().__init__(
            units=units,
            max_order=max_order,
            input_clip=input_clip,
            **kwargs,
        )

    def build(self, input_shape):
        """Build the layer with optional trainable modulus."""
        if self.learnable_modulus:
            self.k_var = self.add_weight(
                name="modulus",
                shape=(),
                initializer=tf.keras.initializers.Constant(self.initial_modulus),
                constraint=tf.keras.constraints.MinMaxNorm(
                    min_value=0.0, max_value=0.999
                ),
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.k_var = tf.constant(self.initial_modulus, dtype=self.dtype)

        super().build(input_shape)

    def _get_num_basis_functions(self) -> int:
        """Return number of elliptic function basis functions."""
        # sn, cn, dn at different frequencies
        return 3 * self.max_order

    def special_basis(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Compute Jacobi elliptic function basis.

        Uses polynomial approximation that interpolates between
        trigonometric (k=0) and hyperbolic (k=1) limits.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, num_basis).
        """
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        k = tf.cast(self.k_var, tf.float64)
        k2 = k * k  # k² appears frequently

        basis = []

        for n in range(1, self.max_order + 1):
            freq = tf.cast(n, tf.float64)
            u = freq * x

            # Approximate sn(u, k)
            # sn(u, k) ≈ sin(u) - k² sin(u) cos²(u) / 4 + O(k⁴)
            sin_u = tf.sin(u)
            cos_u = tf.cos(u)
            sn_u = sin_u - k2 * sin_u * cos_u * cos_u / 4.0

            # Approximate cn(u, k)
            # cn(u, k) ≈ cos(u) + k² sin²(u) cos(u) / 4 + O(k⁴)
            cn_u = cos_u + k2 * sin_u * sin_u * cos_u / 4.0

            # Approximate dn(u, k)
            # dn(u, k) ≈ 1 - k² sin²(u) / 2 + O(k⁴)
            dn_u = 1.0 - k2 * sin_u * sin_u / 2.0

            # Add hyperbolic correction for larger k
            if self.initial_modulus > 0.7:
                tanh_u = tf.tanh(u)
                sech_u = 1.0 / tf.cosh(u)
                # Blend towards hyperbolic limit
                blend = (k - 0.7) / 0.3
                sn_u = (1.0 - blend) * sn_u + blend * tanh_u
                cn_u = (1.0 - blend) * cn_u + blend * sech_u
                dn_u = (1.0 - blend) * dn_u + blend * sech_u

            basis.extend([sn_u, cn_u, dn_u])

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "modulus": float(self.initial_modulus),
            "learnable_modulus": self.learnable_modulus,
        })
        return config
