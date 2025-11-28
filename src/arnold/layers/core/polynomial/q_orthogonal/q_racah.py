# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
q-Racah polynomials.

This module implements q-Racah polynomials :math:`R_n(x; \alpha, \beta, \gamma, \delta; q)`,
the top of the Askey-Wilson scheme for discrete q-orthogonal polynomials.

Mathematical Background
-----------------------
The q-Racah polynomials are defined for :math:`n = 0, 1, \ldots, N` where one of
:math:`\alpha q, \beta\delta q, \gamma q = q^{-N}`:

.. math::

    R_n(\mu(y); \alpha, \beta, \gamma, \delta | q) =
        {}_4\phi_3\left(\begin{array}{c}
        q^{-n}, \alpha\beta q^{n+1}, q^{-y}, \gamma\delta q^{y+1} \\
        \alpha q, \beta\delta q, \gamma q \end{array}; q, q\right)

where :math:`\mu(y) = q^{-y} + \gamma\delta q^{y+1}`.

Notes
-----
q-Racah polynomials satisfy a duality relation: swapping :math:`(\alpha, \beta) \leftrightarrow (\gamma, \delta)`
and :math:`n \leftrightarrow y` gives the same polynomial value.

References
----------
.. [DLMF] NIST Digital Library of Mathematical Functions, §18.28(viii)
.. [KLS] Koekoek et al. (2010), Chapter 14.2
"""

import tensorflow as tf

from arnold.utils.compilation import kan_function
from arnold.utils.constraints import softplus_lower_bound

from .base import QPolynomialBase, _inverse_softplus_lower_bound


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="QRacah")
class QRacah(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using q-Racah polynomials.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.
    units : int
        Output dimensionality.
    alpha, beta, gamma, delta : float
        Parameters, all :math:`> 0`. Defaults 1.0.
    N : int
        Discrete support size, must be >= degree.
    alpha_trainable, beta_trainable, gamma_trainable, delta_trainable : bool
        Trainability flags.
    q : float
        Base parameter in (0, 1). Default 0.5.

    References
    ----------
    .. [1] NIST DLMF 18.28(viii) - q-Racah Polynomials
    .. [2] Koekoek et al. (2010), Chapter 14.2
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        delta: float = 1.0,
        N: int = 10,
        alpha_trainable: bool = False,
        beta_trainable: bool = False,
        gamma_trainable: bool = False,
        delta_trainable: bool = False,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        if degree > N:
            raise ValueError(f"N must be >= degree, got N={N}, degree={degree}")

        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )

        self.alpha_init = alpha
        self.beta_init = beta
        self.gamma_init = gamma
        self.delta_init = delta
        self.N = N
        self.alpha_trainable = alpha_trainable
        self.beta_trainable = beta_trainable
        self.gamma_trainable = gamma_trainable
        self.delta_trainable = delta_trainable

        self.alpha_logits = None
        self.beta_logits = None
        self.gamma_logits = None
        self.delta_logits = None

    def build(self, input_shape):
        super().build(input_shape)

        alpha_logit = _inverse_softplus_lower_bound(self.alpha_init, 0.0)
        beta_logit = _inverse_softplus_lower_bound(self.beta_init, 0.0)
        gamma_logit = _inverse_softplus_lower_bound(self.gamma_init, 0.0)
        delta_logit = _inverse_softplus_lower_bound(self.delta_init, 0.0)

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
        self.gamma_logits = self.add_weight(
            name="gamma_logits",
            shape=(),
            initializer=tf.constant_initializer(gamma_logit),
            trainable=self.gamma_trainable,
            dtype=self.dtype,
        )
        self.delta_logits = self.add_weight(
            name="delta_logits",
            shape=(),
            initializer=tf.constant_initializer(delta_logit),
            trainable=self.delta_trainable,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        """Compute q-Racah polynomial basis using three-term recurrence."""
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)

        q = self._get_q(tf.float64)
        alpha = softplus_lower_bound(tf.cast(self.alpha_logits, tf.float64), 0.0)
        beta = softplus_lower_bound(tf.cast(self.beta_logits, tf.float64), 0.0)
        gamma = softplus_lower_bound(tf.cast(self.gamma_logits, tf.float64), 0.0)
        delta = softplus_lower_bound(tf.cast(self.delta_logits, tf.float64), 0.0)
        N = tf.cast(self.N, tf.float64)

        ones = tf.ones_like(x, dtype=tf.float64)

        # R_0(x) = 1
        R0 = ones
        basis = [R0]

        if self.degree >= 1:
            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for q-Racah."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)
                ab_q = alpha * beta * q
                gd_q = gamma * delta * q

                # Denominator terms
                denom1 = 1.0 - ab_q * tf.pow(q, 2.0 * n_f + 1.0)
                denom2 = 1.0 - ab_q * tf.pow(q, 2.0 * n_f + 2.0)
                denom3 = 1.0 - ab_q * tf.pow(q, 2.0 * n_f)

                # A_n (coefficient of R_{n+1})
                numer_A = (1.0 - alpha * q_n * q) * (1.0 - beta * delta * q_n * q) * \
                          (1.0 - gamma * q_n * q) * (1.0 - ab_q * tf.pow(q, n_f + N + 1.0))
                A_n = numer_A / (denom1 * denom2 + 1e-12)

                # C_n (coefficient of R_{n-1})
                numer_C = (1.0 - q_n) * (alpha - gamma * q_n) * \
                          (beta * delta - gamma * q_n) * \
                          (1.0 - gd_q * tf.pow(q, n_f + N))
                C_n = -alpha * gamma * q * numer_C / ((denom1 * denom3 + 1e-12) * ab_q)

                # B_n = (1 + gd*q) - A_n - C_n
                B_n = 1.0 + gd_q - A_n - C_n

                return A_n, B_n, C_n

            # R_1(x)
            A0, B0, _ = compute_coeffs(0)
            R1 = (x - B0 * ones) / (A0 + 1e-12)
            basis.append(R1)

            # R_n for n >= 2
            R_prev = R0
            R_curr = R1

            for n in range(1, self.degree):
                A_n, B_n, C_n = compute_coeffs(n)
                R_next = (x * R_curr - B_n * R_curr - C_n * R_prev) / (A_n + 1e-12)
                basis.append(R_next)
                R_prev = R_curr
                R_curr = R_next

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha_init,
            "beta": self.beta_init,
            "gamma": self.gamma_init,
            "delta": self.delta_init,
            "N": self.N,
            "alpha_trainable": self.alpha_trainable,
            "beta_trainable": self.beta_trainable,
            "gamma_trainable": self.gamma_trainable,
            "delta_trainable": self.delta_trainable,
        })
        return config


__all__ = [
    "QRacah",
]
