# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
q-Hahn polynomials.

This module implements q-Hahn polynomials :math:`Q_n(x; \alpha, \beta, N; q)`,
the q-analog of classical Hahn polynomials.

Mathematical Background
-----------------------
The q-Hahn polynomials are defined as:

.. math::

    Q_n(x) = Q_n(x; \alpha, \beta, N; q) = 
        {}_3\phi_2\left(\begin{array}{c} q^{-n}, \alpha\beta q^{n+1}, x \\
        \alpha q, q^{-N} \end{array}; q, q\right)

for :math:`n = 0, 1, \ldots, N`.

In the limit :math:`q \to 1`, q-Hahn polynomials reduce to classical Hahn polynomials.

References
----------
.. [DLMF] NIST Digital Library of Mathematical Functions, §18.27(ii)
   https://dlmf.nist.gov/18.27
"""

import tensorflow as tf

from arnold.utils.compilation import kan_function
from arnold.utils.constraints import softplus_lower_bound

from .base import QPolynomialBase, _inverse_softplus_lower_bound


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="QHahn")
class QHahn(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using q-Hahn polynomials.

    The q-Hahn polynomials :math:`Q_n(x; \alpha, \beta, N; q)` are defined as:

    .. math::

        Q_n(x) = Q_n(x; \alpha, \beta, N; q) = 
            {}_3\phi_2\left(\begin{array}{c} q^{-n}, \alpha\beta q^{n+1}, x \\
            \alpha q, q^{-N} \end{array}; q, q\right)

    for :math:`n = 0, 1, \ldots, N`.

    The orthogonality relation is:

    .. math::

        \sum_{y=0}^{N} Q_n(q^{-y}) Q_m(q^{-y}) \binom{N}{y}_q 
            \frac{(\alpha q; q)_y (\beta q; q)_{N-y}}{(\alpha\beta q^2; q)_N} 
            (\alpha q)^y = h_n \delta_{n,m}

    Three-term recurrence:

    .. math::

        x Q_n(x) = A_n Q_{n+1}(x) + B_n Q_n(x) + C_n Q_{n-1}(x)

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.
    units : int
        Output dimensionality.
    alpha : float
        Parameter :math:`\alpha > 0` or :math:`\alpha < -q^{-1}`. Default 1.0.
    beta : float
        Parameter :math:`\beta > 0` or :math:`\beta < -q^{-1}`. Default 1.0.
    N : int
        Discrete support size, must be >= degree.
    alpha_trainable : bool
        Whether :math:`\alpha` is trainable. Default False.
    beta_trainable : bool
        Whether :math:`\beta` is trainable. Default False.
    q : float
        Base parameter in (0, 1). Default 0.5.
    q_trainable : bool
        Whether q is trainable. Default True.

    Notes
    -----
    In the limit :math:`q \to 1`, q-Hahn polynomials reduce to classical Hahn polynomials.

    References
    ----------
    .. [1] NIST DLMF 18.27(ii) - q-Hahn Polynomials
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        N: int = 10,
        alpha_trainable: bool = False,
        beta_trainable: bool = False,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        if N < degree:
            raise ValueError(f"N must be >= degree, got N={N}, degree={degree}")

        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )

        self.alpha_init = alpha
        self.beta_init = beta
        self.N = N
        self.alpha_trainable = alpha_trainable
        self.beta_trainable = beta_trainable

        self.alpha_logits = None
        self.beta_logits = None

    def build(self, input_shape):
        super().build(input_shape)

        # α, β > 0 via softplus
        alpha_logit = _inverse_softplus_lower_bound(self.alpha_init, 0.0)
        beta_logit = _inverse_softplus_lower_bound(self.beta_init, 0.0)

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
        """Compute q-Hahn polynomial basis using three-term recurrence."""
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)

        q = self._get_q(tf.float64)
        alpha = softplus_lower_bound(tf.cast(self.alpha_logits, tf.float64), 0.0)
        beta = softplus_lower_bound(tf.cast(self.beta_logits, tf.float64), 0.0)
        N = tf.cast(self.N, tf.float64)

        ones = tf.ones_like(x, dtype=tf.float64)

        # Q_0(x) = 1
        Q0 = ones
        basis = [Q0]

        if self.degree >= 1:
            # Three-term recurrence coefficients for q-Hahn
            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for q-Hahn."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)

                # Numerator/denominator terms
                ab_q = alpha * beta * q

                # A_n coefficient (for Q_{n+1})
                denom1 = (1.0 - ab_q * tf.pow(q, 2.0 * n_f + 1.0))
                denom2 = (1.0 - ab_q * tf.pow(q, 2.0 * n_f + 2.0))

                A_n = (1.0 - q_n * tf.pow(q, 1.0)) * (1.0 - alpha * q_n * q) * \
                      (1.0 - beta * q_n * q) * (1.0 - tf.pow(q, n_f - N))
                A_n = A_n / (denom1 * denom2 + 1e-12)

                # C_n coefficient (for Q_{n-1})
                C_n = -alpha * beta * q * q_n * (1.0 - q_n) * \
                      (1.0 - ab_q * tf.pow(q, n_f + N)) * \
                      (1.0 - ab_q * tf.pow(q, n_f))
                denom3 = (1.0 - ab_q * tf.pow(q, 2.0 * n_f))
                C_n = C_n / (denom1 * denom3 + 1e-12)

                # B_n = 1 - A_n - C_n for normalized form
                B_n = 1.0 - A_n - C_n

                return A_n, B_n, C_n

            # Q_1: Use simplified first step
            A0, B0, C0 = compute_coeffs(0)
            Q1 = (x - B0 * ones) / (A0 + 1e-12)
            basis.append(Q1)

            # Q_n for n >= 2
            Q_prev = Q0
            Q_curr = Q1

            for n in range(1, self.degree):
                A_n, B_n, C_n = compute_coeffs(n)
                # x Q_n = A_n Q_{n+1} + B_n Q_n + C_n Q_{n-1}
                # => Q_{n+1} = (x Q_n - B_n Q_n - C_n Q_{n-1}) / A_n
                Q_next = (x * Q_curr - B_n * Q_curr - C_n * Q_prev) / (A_n + 1e-12)
                basis.append(Q_next)
                Q_prev = Q_curr
                Q_curr = Q_next

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha_init,
            "beta": self.beta_init,
            "N": self.N,
            "alpha_trainable": self.alpha_trainable,
            "beta_trainable": self.beta_trainable,
        })
        return config


__all__ = [
    "QHahn",
]
