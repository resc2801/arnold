# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Parabolic cylinder function KAN layer.

This module provides :class:`ParabolicCylinder`, a KAN layer using parabolic
cylinder functions (Weber functions) as basis functions.

Mathematical Background
-----------------------
Parabolic cylinder functions are solutions to the parabolic cylinder
differential equation:

.. math::

    y'' + \left(\nu + \frac{1}{2} - \frac{x^2}{4}\right) y = 0

The standard solutions are:

- :math:`D_\nu(x)`: Parabolic cylinder function (Whittaker's notation)
- :math:`U(a, x)`: Parabolic cylinder function (NIST notation)

These functions arise in:
- Quantum harmonic oscillator (Hermite functions)
- Diffraction theory
- Probability theory (error function related)

Connection to Hermite Polynomials
---------------------------------
For non-negative integer n:

.. math::

    D_n(x) = 2^{-n/2} e^{-x^2/4} H_n\left(\frac{x}{\sqrt{2}}\right)

where :math:`H_n(x)` are Hermite polynomials.

References
----------
.. [1] NIST DLMF, Ch. 12: Parabolic Cylinder Functions
.. [2] Abramowitz & Stegun, Ch. 19: Parabolic Cylinder Functions
"""
import tensorflow as tf

from arnold.layers.core.special.base import SpecialBase


tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold", name="ParabolicCylinder")
class ParabolicCylinder(SpecialBase):
    r"""
    Kolmogorov-Arnold Network layer using parabolic cylinder functions.

    For integer orders, these are related to Hermite functions
    (Hermite polynomials × Gaussian weight).

    Parameters
    ----------
    units : int
        Output dimension.
    max_order : int, default=8
        Maximum order n for D_n(x).
    input_clip : tuple[float, float] | None, default=(-10.0, 10.0)
        Input clipping for numerical stability.
    **kwargs
        Additional Keras layer arguments.

    Example
    -------
    >>> import tensorflow as tf
    >>> from arnold.layers.core.special import ParabolicCylinder
    >>>
    >>> layer = ParabolicCylinder(max_order=6, units=32)
    >>> x = tf.random.uniform((16, 10), -5, 5)
    >>> y = layer(x)  # shape: (16, 32)
    """

    def __init__(
        self,
        *,
        units: int,
        max_order: int = 8,
        input_clip: tuple[float, float] | None = (-10.0, 10.0),
        **kwargs,
    ):
        super().__init__(
            units=units,
            max_order=max_order,
            input_clip=input_clip,
            **kwargs,
        )

    def _get_num_basis_functions(self) -> int:
        """Return number of parabolic cylinder basis functions."""
        return self.max_order + 1

    def special_basis(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Compute parabolic cylinder function basis.

        For integer n, uses the relation to Hermite polynomials:

        .. math::

            D_n(x) = 2^{-n/2} e^{-x^2/4} H_n(x/\sqrt{2})

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

        # Gaussian weight
        gaussian_weight = tf.exp(-x * x / 4.0)

        # Scaled argument for Hermite polynomials
        xi = x / tf.sqrt(2.0)

        # Compute Hermite polynomials using recurrence
        # H_0(x) = 1, H_1(x) = 2x
        # H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
        H0 = tf.ones_like(xi, dtype=tf.float64)
        H1 = 2.0 * xi

        basis = []

        # D_0(x) = e^{-x²/4}
        D0 = gaussian_weight * H0
        basis.append(D0)

        if self.max_order >= 1:
            # D_1(x) = 2^{-1/2} e^{-x²/4} H_1(x/√2)
            D1 = gaussian_weight * H1 * tf.pow(2.0, -0.5)
            basis.append(D1)

            H_prev2 = H0
            H_prev1 = H1

            for n in range(2, self.max_order + 1):
                n_f = tf.cast(n, tf.float64)
                # Hermite recurrence: H_{n}(x) = 2x H_{n-1}(x) - 2(n-1) H_{n-2}(x)
                H_n = 2.0 * xi * H_prev1 - 2.0 * (n_f - 1.0) * H_prev2
                # D_n(x) = 2^{-n/2} e^{-x²/4} H_n(x/√2)
                D_n = gaussian_weight * H_n * tf.pow(2.0, -n_f / 2.0)
                basis.append(D_n)
                H_prev2 = H_prev1
                H_prev1 = H_n

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)
