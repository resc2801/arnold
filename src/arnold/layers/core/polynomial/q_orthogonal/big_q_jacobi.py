# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Big q-Jacobi polynomials.

This module implements Big q-Jacobi polynomials :math:`P_n(x; a, b, c; q)`.

Mathematical Background
-----------------------
The Big q-Jacobi polynomials are defined as:

.. math::

    P_n(x; a, b, c; q) = {}_3\phi_2\left(\begin{array}{c} 
        q^{-n}, abq^{n+1}, x \\ aq, cq \end{array}; q, q\right)

In the limit :math:`q \to 1`, Big q-Jacobi polynomials reduce to Jacobi polynomials.

References
----------
.. [DLMF] NIST Digital Library of Mathematical Functions, §18.27(iii)
   https://dlmf.nist.gov/18.27
"""

import tensorflow as tf

from arnold.utils.compilation import kan_function
from arnold.utils.constraints import softplus_lower_bound

from .base import QPolynomialBase, _inverse_softplus_lower_bound


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="BigQJacobi")
class BigQJacobi(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Big q-Jacobi polynomials.

    The Big q-Jacobi polynomials :math:`P_n(x; a, b, c; q)` are defined as:

    .. math::

        P_n(x; a, b, c; q) = {}_3\phi_2\left(\begin{array}{c} 
            q^{-n}, abq^{n+1}, x \\ aq, cq \end{array}; q, q\right)

    They are orthogonal with respect to a discrete measure on 
    :math:`\{aq^{k+1}\}_{k=0}^{\infty} \cup \{cq^{k+1}\}_{k=0}^{\infty}`.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.
    units : int
        Output dimensionality.
    a : float
        Parameter, :math:`0 < a < q^{-1}`. Default 0.5.
    b : float
        Parameter, :math:`0 < b < q^{-1}`. Default 0.5.
    c : float
        Parameter, :math:`c < 0`. Default -0.5.
    a_trainable, b_trainable, c_trainable : bool
        Trainability flags.
    q : float
        Base parameter in (0, 1). Default 0.5.

    References
    ----------
    .. [1] NIST DLMF 18.27(iii) - Big q-Jacobi Polynomials
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        a: float = 0.5,
        b: float = 0.5,
        c: float = -0.5,
        a_trainable: bool = False,
        b_trainable: bool = False,
        c_trainable: bool = False,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )

        self.a_init = a
        self.b_init = b
        self.c_init = c
        self.a_trainable = a_trainable
        self.b_trainable = b_trainable
        self.c_trainable = c_trainable

        self.a_logits = None
        self.b_logits = None
        self.c_logits = None

    def build(self, input_shape):
        super().build(input_shape)

        # a, b > 0 via softplus; c < 0 via negative softplus
        a_logit = _inverse_softplus_lower_bound(self.a_init, 0.0)
        b_logit = _inverse_softplus_lower_bound(self.b_init, 0.0)
        # For c < 0, we store |c| and negate in forward pass
        c_logit = _inverse_softplus_lower_bound(abs(self.c_init), 0.0)

        self.a_logits = self.add_weight(
            name="a_logits",
            shape=(),
            initializer=tf.constant_initializer(a_logit),
            trainable=self.a_trainable,
            dtype=self.dtype,
        )
        self.b_logits = self.add_weight(
            name="b_logits",
            shape=(),
            initializer=tf.constant_initializer(b_logit),
            trainable=self.b_trainable,
            dtype=self.dtype,
        )
        self.c_logits = self.add_weight(
            name="c_logits",
            shape=(),
            initializer=tf.constant_initializer(c_logit),
            trainable=self.c_trainable,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        """Compute Big q-Jacobi polynomial basis using three-term recurrence."""
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)

        q = self._get_q(tf.float64)
        a = softplus_lower_bound(tf.cast(self.a_logits, tf.float64), 0.0)
        b = softplus_lower_bound(tf.cast(self.b_logits, tf.float64), 0.0)
        c = -softplus_lower_bound(tf.cast(self.c_logits, tf.float64), 0.0)  # c < 0

        ones = tf.ones_like(x, dtype=tf.float64)

        # P_0(x) = 1
        P0 = ones
        basis = [P0]

        if self.degree >= 1:
            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for Big q-Jacobi."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)
                ab_q = a * b * q

                # A_n (coefficient of P_{n+1})
                numer_A = (1.0 - a * q_n * q) * (1.0 - c * q_n * q) * \
                          (1.0 - ab_q * tf.pow(q, 2.0 * n_f + 1.0))
                denom_A = (1.0 - ab_q * tf.pow(q, 2.0 * n_f + 1.0)) * \
                          (1.0 - ab_q * tf.pow(q, 2.0 * n_f + 2.0))
                A_n = numer_A / (denom_A + 1e-12)

                # C_n (coefficient of P_{n-1})
                numer_C = -a * c * q * q_n * (1.0 - q_n) * (1.0 - b * q_n) * \
                          (1.0 - ab_q * q_n / c)
                denom_C = (1.0 - ab_q * tf.pow(q, 2.0 * n_f)) * \
                          (1.0 - ab_q * tf.pow(q, 2.0 * n_f + 1.0))
                C_n = numer_C / (denom_C + 1e-12)

                # B_n (coefficient of P_n) - from normalization
                B_n = a * q + c * q - A_n - C_n

                return A_n, B_n, C_n

            # P_1(x)
            A0, B0, C0 = compute_coeffs(0)
            P1 = (x - B0 * ones) / (A0 + 1e-12)
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
            "a": self.a_init,
            "b": self.b_init,
            "c": self.c_init,
            "a_trainable": self.a_trainable,
            "b_trainable": self.b_trainable,
            "c_trainable": self.c_trainable,
        })
        return config


__all__ = [
    "BigQJacobi",
]
