# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Continuous q-Jacobi polynomials.

This module implements Continuous q-Jacobi polynomials :math:`P_n^{(\alpha,\beta)}(x | q)`.

Mathematical Background
-----------------------
The Continuous q-Jacobi polynomials are defined as:

.. math::

    P_n^{(\alpha,\beta)}(x | q) = \frac{(q^{\alpha+1}; q)_n}{(q; q)_n} \,
        {}_4\phi_3\left(\begin{array}{c}
        q^{-n}, q^{n+\alpha+\beta+1}, q^{\alpha/2+1/4} e^{i\theta}, q^{\alpha/2+1/4} e^{-i\theta} \\
        q^{\alpha+1}, -q^{(\alpha+\beta+1)/2}, -q^{(\alpha+\beta+2)/2}
        \end{array}; q, q\right)

where :math:`x = \cos\theta`.

Properties
----------
- Limit: :math:`\lim_{q \to 1} P_n^{(\alpha,\beta)}(x | q) = P_n^{(\alpha,\beta)}(x)`
- Special case: :math:`P_n^{(\lambda-1/2, \lambda-1/2)}(x | q) \propto C_n(x; q^\lambda | q)`
- Orthogonal on [-1, 1] with q-deformed beta weight

References
----------
.. [1] NIST DLMF 18.28(ix)
.. [2] Koekoek et al. (2010), Chapter 14.10
"""

import tensorflow as tf

from arnold.utils.compilation import kan_function
from arnold.utils.constraints import softplus_lower_bound

from .base import QPolynomialBase, _inverse_softplus_lower_bound


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="ContinuousQJacobi")
class ContinuousQJacobi(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Continuous q-Jacobi polynomials.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.
    units : int
        Output dimensionality.
    alpha : float
        Parameter α > -1. Default 0.0.
    beta : float
        Parameter β > -1. Default 0.0.
    alpha_trainable : bool
        Whether α is trainable.
    beta_trainable : bool
        Whether β is trainable.
    q : float
        Base parameter in (0, 1). Default 0.5.

    References
    ----------
    .. [1] NIST DLMF 18.28(ix)
    .. [2] Koekoek et al. (2010), Chapter 14.10
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        alpha: float = 0.0,
        beta: float = 0.0,
        alpha_trainable: bool = False,
        beta_trainable: bool = False,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        **kwargs,
    ):
        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )

        self.alpha_init = alpha
        self.beta_init = beta
        self.alpha_trainable = alpha_trainable
        self.beta_trainable = beta_trainable

        self.alpha_logits = None
        self.beta_logits = None

    def build(self, input_shape):
        super().build(input_shape)

        # α, β > -1 via softplus with lower bound -1
        alpha_logit = _inverse_softplus_lower_bound(self.alpha_init + 1.0, 0.0)
        beta_logit = _inverse_softplus_lower_bound(self.beta_init + 1.0, 0.0)

        self.alpha_logits = self.add_weight(
            name="alpha_logits",
            shape=(),
            initializer=tf.constant_initializer(alpha_logit),
            trainable=self.alpha_trainable,
            dtype=self.dtype,
        )
        self.beta_logits = self.add_weight(
            name="beta_logits",
            shape=(),
            initializer=tf.constant_initializer(beta_logit),
            trainable=self.beta_trainable,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Compute Continuous q-Jacobi polynomial basis using three-term recurrence.

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
        # α, β > -1: we store α+1, β+1 > 0
        alpha = softplus_lower_bound(tf.cast(self.alpha_logits, tf.float64), 0.0) - 1.0
        beta = softplus_lower_bound(tf.cast(self.beta_logits, tf.float64), 0.0) - 1.0

        # P_0(x) = 1
        P0 = tf.ones_like(x, dtype=tf.float64)
        basis = [P0]

        if self.degree >= 1:
            # Three-term recurrence coefficients for continuous q-Jacobi
            # Simplified form based on Koekoek et al.

            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for continuous q-Jacobi."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)

                ab = alpha + beta
                q_ab = tf.pow(q, ab)
                q_a = tf.pow(q, alpha)
                q_b = tf.pow(q, beta)

                # Denominator terms
                denom1 = 1.0 - q_ab * tf.pow(q, 2.0 * n_f + 1.0)
                denom2 = 1.0 - q_ab * tf.pow(q, 2.0 * n_f + 2.0)
                denom3 = 1.0 - q_ab * tf.pow(q, 2.0 * n_f)

                # A_n (coefficient of P_{n+1})
                numer_A = (1.0 - q_a * q_n * q) * (1.0 - q_ab * q_n * q)
                A_n = 0.5 * numer_A / (denom1 * denom2 + 1e-12)

                # C_n (coefficient of P_{n-1})
                numer_C = (1.0 - q_n) * (1.0 - q_b * q_n)
                C_n = 0.5 * q_a * q * numer_C / (denom1 * denom3 + 1e-12)

                # B_n
                B_n = 0.5 * (1.0 + q_a * q_b * q) - A_n - C_n

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
            "beta": self.beta_init,
            "alpha_trainable": self.alpha_trainable,
            "beta_trainable": self.beta_trainable,
        })
        return config


__all__ = [
    "ContinuousQJacobi",
]
