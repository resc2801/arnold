# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Little q-Jacobi polynomials.

This module implements Little q-Jacobi polynomials :math:`p_n(x; a, b; q)`.

Mathematical Background
-----------------------
The Little q-Jacobi polynomials are defined as:

.. math::

    p_n(x; a, b; q) = {}_2\phi_1\left(\begin{array}{c} 
        q^{-n}, abq^{n+1} \\ aq \end{array}; q, qx\right)

They are orthogonal with respect to a discrete measure on :math:`\{q^k\}_{k=0}^{\infty}`.

Notes
-----
Little q-Jacobi polynomials with :math:`b=0` are called Little q-Laguerre or Wall polynomials.

References
----------
.. [DLMF] NIST Digital Library of Mathematical Functions, §18.27(iv)
   https://dlmf.nist.gov/18.27
"""

import tensorflow as tf

from arnold.utils.compilation import kan_function
from arnold.utils.constraints import softplus_lower_bound

from .base import QPolynomialBase, _inverse_softplus_lower_bound


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="LittleQJacobi")
class LittleQJacobi(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Little q-Jacobi polynomials.

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
    a_trainable, b_trainable : bool
        Trainability flags.
    q : float
        Base parameter in (0, 1). Default 0.5.

    References
    ----------
    .. [1] NIST DLMF 18.27(iv) - Little q-Jacobi Polynomials
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        a: float = 0.5,
        b: float = 0.5,
        a_trainable: bool = False,
        b_trainable: bool = False,
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
        self.a_trainable = a_trainable
        self.b_trainable = b_trainable

        self.a_logits = None
        self.b_logits = None

    def build(self, input_shape):
        super().build(input_shape)

        a_logit = _inverse_softplus_lower_bound(self.a_init, 0.0)
        b_logit = _inverse_softplus_lower_bound(self.b_init, 0.0)

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

    @kan_fn
    def pseudo_vandermonde(self, x):
        """Compute Little q-Jacobi polynomial basis using three-term recurrence."""
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)

        q = self._get_q(tf.float64)
        a = softplus_lower_bound(tf.cast(self.a_logits, tf.float64), 0.0)
        b = softplus_lower_bound(tf.cast(self.b_logits, tf.float64), 0.0)

        ones = tf.ones_like(x, dtype=tf.float64)

        # p_0(x) = 1
        p0 = ones
        basis = [p0]

        if self.degree >= 1:
            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for Little q-Jacobi."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)
                ab_q = a * b * q

                # A_n
                numer_A = (1.0 - a * q_n * q) * (1.0 - ab_q * tf.pow(q, 2.0 * n_f + 1.0))
                denom_A = (1.0 - ab_q * tf.pow(q, 2.0 * n_f + 1.0)) * \
                          (1.0 - ab_q * tf.pow(q, 2.0 * n_f + 2.0))
                A_n = numer_A / (denom_A + 1e-12)

                # C_n
                numer_C = a * q * q_n * (1.0 - q_n) * (1.0 - b * q_n)
                denom_C = (1.0 - ab_q * tf.pow(q, 2.0 * n_f)) * \
                          (1.0 - ab_q * tf.pow(q, 2.0 * n_f + 1.0))
                C_n = numer_C / (denom_C + 1e-12)

                # B_n
                B_n = a * q - A_n - C_n

                return A_n, B_n, C_n

            # p_1(x)
            A0, B0, C0 = compute_coeffs(0)
            p1 = (x - B0 * ones) / (A0 + 1e-12)
            basis.append(p1)

            # p_n for n >= 2
            p_prev = p0
            p_curr = p1

            for n in range(1, self.degree):
                A_n, B_n, C_n = compute_coeffs(n)
                p_next = (x * p_curr - B_n * p_curr - C_n * p_prev) / (A_n + 1e-12)
                basis.append(p_next)
                p_prev = p_curr
                p_curr = p_next

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "a": self.a_init,
            "b": self.b_init,
            "a_trainable": self.a_trainable,
            "b_trainable": self.b_trainable,
        })
        return config


__all__ = [
    "LittleQJacobi",
]
