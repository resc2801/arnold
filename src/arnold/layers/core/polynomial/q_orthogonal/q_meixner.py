# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
q-Meixner polynomials.

This module implements q-Meixner polynomials :math:`M_n(x; b, c; q)`.

Mathematical Background
-----------------------
The q-Meixner polynomials are the q-analog of classical Meixner polynomials:

.. math::

    M_n(q^{-x}; b, c; q) = {}_2\phi_1\left(\begin{array}{c} 
        q^{-n}, q^{-x} \\ bq \end{array}; q, -\frac{q^{n+1}}{c}\right)

They are orthogonal on :math:`x \in \{0, 1, 2, \ldots\}` with a q-negative binomial weight.

References
----------
.. [KLS] Koekoek et al. (2010), Chapter 14.11
"""

import tensorflow as tf

from arnold.utils.compilation import kan_function
from arnold.utils.constraints import softplus_lower_bound

from .base import QPolynomialBase, _inverse_softplus_lower_bound


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="QMeixner")
class QMeixner(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using q-Meixner polynomials.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.
    units : int
        Output dimensionality.
    b : float
        Parameter :math:`b > 0`. Default 1.0.
    c : float
        Parameter :math:`c > 0`. Default 0.5.
    b_trainable, c_trainable : bool
        Trainability flags.
    q : float
        Base parameter in (0, 1). Default 0.5.

    Notes
    -----
    In the limit :math:`q \to 1`, q-Meixner polynomials reduce to classical Meixner polynomials.

    References
    ----------
    .. [1] Koekoek et al. (2010), Chapter 14.11
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        b: float = 1.0,
        c: float = 0.5,
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

        self.b_init = b
        self.c_init = c
        self.b_trainable = b_trainable
        self.c_trainable = c_trainable

        self.b_logits = None
        self.c_logits = None

    def build(self, input_shape):
        super().build(input_shape)

        b_logit = _inverse_softplus_lower_bound(self.b_init, 0.0)
        c_logit = _inverse_softplus_lower_bound(self.c_init, 0.0)

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
        """Compute q-Meixner polynomial basis using three-term recurrence."""
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)

        q = self._get_q(tf.float64)
        b = softplus_lower_bound(tf.cast(self.b_logits, tf.float64), 0.0)
        c = softplus_lower_bound(tf.cast(self.c_logits, tf.float64), 0.0)

        ones = tf.ones_like(x, dtype=tf.float64)

        # M_0(x) = 1
        M0 = ones
        basis = [M0]

        if self.degree >= 1:
            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for q-Meixner."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)

                # A_n (coefficient of M_{n+1})
                A_n = -c * (1.0 - b * q_n * q) / (1.0 + c)

                # C_n (coefficient of M_{n-1})
                C_n = (1.0 - q_n) * (1.0 + c * q_n) / (1.0 + c)

                # B_n
                B_n = 1.0 + c * b * q - A_n - C_n

                return A_n, B_n, C_n

            # M_1(x)
            A0, B0, C0 = compute_coeffs(0)
            M1 = (x - B0 * ones) / (A0 + 1e-12)
            basis.append(M1)

            # M_n for n >= 2
            M_prev = M0
            M_curr = M1

            for n in range(1, self.degree):
                A_n, B_n, C_n = compute_coeffs(n)
                M_next = (x * M_curr - B_n * M_curr - C_n * M_prev) / (A_n + 1e-12)
                basis.append(M_next)
                M_prev = M_curr
                M_curr = M_next

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "b": self.b_init,
            "c": self.c_init,
            "b_trainable": self.b_trainable,
            "c_trainable": self.c_trainable,
        })
        return config


__all__ = [
    "QMeixner",
]
