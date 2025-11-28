# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
q-Charlier polynomials.

This module implements q-Charlier polynomials :math:`C_n(x; a; q)`.

Mathematical Background
-----------------------
The q-Charlier polynomials are the q-analog of classical Charlier polynomials:

.. math::

    C_n(q^{-x}; a; q) = {}_2\phi_1\left(\begin{array}{c}
        q^{-n}, q^{-x} \\ 0 \end{array}; q, -\frac{q^{n+1}}{a}\right)

They are orthogonal on :math:`x \in \{0, 1, 2, \ldots\}` with a q-Poisson weight.

Notes
-----
q-Charlier polynomials are a limit case of q-Meixner with :math:`b \to 0`.

References
----------
.. [KLS] Koekoek et al. (2010), Chapter 14.12
"""

import tensorflow as tf

from arnold.utils.compilation import kan_function
from arnold.utils.constraints import softplus_lower_bound

from .base import QPolynomialBase, _inverse_softplus_lower_bound


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="QCharlier")
class QCharlier(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using q-Charlier polynomials.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.
    units : int
        Output dimensionality.
    a : float
        Parameter :math:`a > 0`. Default 1.0.
    a_trainable : bool
        Whether a is trainable.
    q : float
        Base parameter in (0, 1). Default 0.5.

    References
    ----------
    .. [1] Koekoek et al. (2010), Chapter 14.12
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        a: float = 1.0,
        a_trainable: bool = False,
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
        self.a_trainable = a_trainable
        self.a_logits = None

    def build(self, input_shape):
        super().build(input_shape)

        a_logit = _inverse_softplus_lower_bound(self.a_init, 0.0)

        self.a_logits = self.add_weight(
            name="a_logits",
            shape=(),
            initializer=tf.constant_initializer(a_logit),
            trainable=self.a_trainable,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        """Compute q-Charlier polynomial basis using three-term recurrence."""
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)

        q = self._get_q(tf.float64)
        a = softplus_lower_bound(tf.cast(self.a_logits, tf.float64), 0.0)

        ones = tf.ones_like(x, dtype=tf.float64)

        # C_0(x) = 1
        C0 = ones
        basis = [C0]

        if self.degree >= 1:
            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for q-Charlier."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)

                # A_n (coefficient of C_{n+1})
                A_n = -a * q_n * q

                # C_n (coefficient of C_{n-1})
                C_coeff = (1.0 - q_n)

                # B_n = 1 + a*q - A_n - C_n
                B_n = 1.0 + a * q - A_n - C_coeff

                return A_n, B_n, C_coeff

            # C_1(x)
            A0, B0, _ = compute_coeffs(0)
            C1 = (x - B0 * ones) / (A0 + 1e-12)
            basis.append(C1)

            # C_n for n >= 2
            C_prev = C0
            C_curr = C1

            for n in range(1, self.degree):
                A_n, B_n, C_coeff = compute_coeffs(n)
                C_next = (x * C_curr - B_n * C_curr - C_coeff * C_prev) / (A_n + 1e-12)
                basis.append(C_next)
                C_prev = C_curr
                C_curr = C_next

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "a": self.a_init,
            "a_trainable": self.a_trainable,
        })
        return config


__all__ = [
    "QCharlier",
]
