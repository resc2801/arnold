# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Continuous q-Laguerre polynomials.

This module implements Continuous q-Laguerre polynomials :math:`P_n^{(\alpha)}(x | q)`.

Mathematical Background
-----------------------
The Continuous q-Laguerre polynomials are q-analogs of classical Laguerre polynomials,
orthogonal on :math:`[0, \infty)`.

They satisfy a three-term recurrence:

.. math::

    x P_n(x) = A_n P_{n+1}(x) + B_n P_n(x) + C_n P_{n-1}(x)

where the coefficients depend on α and q.

Properties
----------
- Limit: :math:`\lim_{q \to 1} P_n^{(\alpha)}(x(1-q) | q) = L_n^{(\alpha)}(x)`
- Related to little q-Jacobi polynomials via limiting case
- Applications in q-harmonic analysis and quantum mechanics

References
----------
.. [1] Koekoek et al. (2010), Chapter 14.21
.. [2] Moak (1981). "The q-analogue of the Laguerre polynomials"
"""

import tensorflow as tf

from arnold.utils.compilation import kan_function
from arnold.utils.constraints import softplus_lower_bound

from .base import QPolynomialBase, _inverse_softplus_lower_bound


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="ContinuousQLaguerre")
class ContinuousQLaguerre(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Continuous q-Laguerre polynomials.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.
    units : int
        Output dimensionality.
    alpha : float
        Parameter α > -1. Default 0.0.
    alpha_trainable : bool
        Whether α is trainable.
    q : float
        Base parameter in (0, 1). Default 0.5.

    References
    ----------
    .. [1] Koekoek et al. (2010), Chapter 14.21
    .. [2] Moak (1981)
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        alpha: float = 0.0,
        alpha_trainable: bool = False,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = (0.0, 2.0),
        **kwargs,
    ):
        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )

        self.alpha_init = alpha
        self.alpha_trainable = alpha_trainable
        self.alpha_logits = None

    def build(self, input_shape):
        super().build(input_shape)

        # α > -1 via softplus with lower bound
        alpha_logit = _inverse_softplus_lower_bound(self.alpha_init + 1.0, 0.0)

        self.alpha_logits = self.add_weight(
            name="alpha_logits",
            shape=(),
            initializer=tf.constant_initializer(alpha_logit),
            trainable=self.alpha_trainable,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Compute Continuous q-Laguerre polynomial basis using three-term recurrence.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, degree+1).
        """
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)

        q = self._get_q(tf.float64)
        alpha = softplus_lower_bound(tf.cast(self.alpha_logits, tf.float64), 0.0) - 1.0

        # P_0(x) = 1
        P0 = tf.ones_like(x, dtype=tf.float64)
        basis = [P0]

        if self.degree >= 1:
            # Three-term recurrence for Continuous q-Laguerre
            q_a = tf.pow(q, alpha)

            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for Continuous q-Laguerre."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)

                # A_n (coefficient of P_{n+1})
                A_n = -q_n * q / (1.0 - q_a * q_n * q * q)

                # C_n (coefficient of P_{n-1})
                C_n = q_a * q_n * (1.0 - q_n) / (1.0 - q_a * q_n * q)

                # B_n
                B_n = 1.0 + q_a * q_n * q - A_n * (1.0 - q_a * q_n * q * q) - C_n

                return A_n, B_n, C_n

            # P_1(x)
            A0, B0, _ = compute_coeffs(0)
            P1 = (x - B0) / (A0 + 1e-12)
            basis.append(P1)

            # P_n for n >= 2
            P_prev = P0
            P_curr = P1

            for n in range(1, self.degree):
                A_n, B_n, C_n = compute_coeffs(n)
                P_next = (x * P_curr - B_n * P_curr - C_n * P_prev) / (A_n + 1e-12)
                basis.append(P_next)
                P_prev = P_curr
                P_curr = P_next

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha_init,
            "alpha_trainable": self.alpha_trainable,
        })
        return config


__all__ = [
    "ContinuousQLaguerre",
]
