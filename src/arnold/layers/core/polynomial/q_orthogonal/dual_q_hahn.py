# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Dual q-Hahn polynomials.

This module implements Dual q-Hahn polynomials :math:`R_n(\mu(x); \gamma, \delta, N; q)`.

Mathematical Background
-----------------------
The Dual q-Hahn polynomials are dual to the q-Hahn polynomials:

.. math::

    R_n(\mu(x); \gamma, \delta, N | q) =
        {}_3\phi_2\left(\begin{array}{c}
        q^{-n}, q^{-x}, \gamma\delta q^{x+1} \\
        \gamma q, q^{-N} \end{array}; q, q\right)

where :math:`\mu(x) = q^{-x} + \gamma\delta q^{x+1}`.

Notes
-----
Dual q-Hahn polynomials are limit cases of q-Racah with :math:`\alpha = 0`.

References
----------
.. [KLS] Koekoek et al. (2010), Chapter 14.7
"""

import tensorflow as tf

from arnold.utils.compilation import kan_function
from arnold.utils.constraints import softplus_lower_bound

from .base import QPolynomialBase, _inverse_softplus_lower_bound


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="DualQHahn")
class DualQHahn(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Dual q-Hahn polynomials.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.
    units : int
        Output dimensionality.
    gamma : float
        Parameter :math:`\gamma > 0`. Default 1.0.
    delta : float
        Parameter :math:`\delta > 0`. Default 1.0.
    N : int
        Discrete support size, must be >= degree.
    gamma_trainable, delta_trainable : bool
        Trainability flags.
    q : float
        Base parameter in (0, 1). Default 0.5.

    References
    ----------
    .. [1] Koekoek et al. (2010), Chapter 14.7
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        gamma: float = 1.0,
        delta: float = 1.0,
        N: int = 10,
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

        self.gamma_init = gamma
        self.delta_init = delta
        self.N = N
        self.gamma_trainable = gamma_trainable
        self.delta_trainable = delta_trainable

        self.gamma_logits = None
        self.delta_logits = None

    def build(self, input_shape):
        super().build(input_shape)

        gamma_logit = _inverse_softplus_lower_bound(self.gamma_init, 0.0)
        delta_logit = _inverse_softplus_lower_bound(self.delta_init, 0.0)

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
        """Compute Dual q-Hahn polynomial basis using three-term recurrence."""
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)

        q = self._get_q(tf.float64)
        gamma = softplus_lower_bound(tf.cast(self.gamma_logits, tf.float64), 0.0)
        delta = softplus_lower_bound(tf.cast(self.delta_logits, tf.float64), 0.0)
        N = tf.cast(self.N, tf.float64)

        ones = tf.ones_like(x, dtype=tf.float64)

        # R_0(x) = 1
        R0 = ones
        basis = [R0]

        if self.degree >= 1:
            gd_q = gamma * delta * q

            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for Dual q-Hahn."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)

                # A_n (coefficient of R_{n+1})
                A_n = (1.0 - gamma * q_n * q) * (1.0 - tf.pow(q, n_f - N))

                # C_n (coefficient of R_{n-1})
                C_n = gamma * q_n * (1.0 - q_n) * (delta - tf.pow(q, n_f - N - 1.0))

                # B_n = 1 + gd*q - A_n - C_n
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
            "gamma": self.gamma_init,
            "delta": self.delta_init,
            "N": self.N,
            "gamma_trainable": self.gamma_trainable,
            "delta_trainable": self.delta_trainable,
        })
        return config


__all__ = [
    "DualQHahn",
]
