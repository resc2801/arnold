# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Discrete q-Hermite I polynomials.

This module implements Discrete q-Hermite I polynomials :math:`h_n(x; q)`.

Mathematical Background
-----------------------
The Discrete q-Hermite I polynomials are defined as:

.. math::

    h_n(x; q) = x^n \cdot {}_2\phi_0\left(\begin{array}{c}
        q^{-n}, q^{-n+1} \\ - \end{array}; q^2, \frac{q^{2n-1}}{x^2}\right)

They satisfy the remarkably simple three-term recurrence:

.. math::

    x \, h_n(x; q) = h_{n+1}(x; q) + (1 - q^n) \, h_{n-1}(x; q)

with initial conditions :math:`h_0(x; q) = 1` and :math:`h_1(x; q) = x`.

Properties
----------
- Symmetric: :math:`h_n(-x; q) = (-1)^n h_n(x; q)`
- Limit: :math:`\lim_{q \to 1} h_n(x(1-q^2)^{1/2}; q) / (1-q^2)^{n/2} = 2^{-n} H_n(x)`

References
----------
.. [1] NIST DLMF 18.27(vii)
.. [2] Koekoek et al. (2010), Chapter 14.28
"""

import tensorflow as tf

from arnold.utils.compilation import kan_function

from .base import QPolynomialBase


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="DiscreteQHermite1")
class DiscreteQHermite1(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Discrete q-Hermite I polynomials.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.
    units : int
        Output dimensionality.
    q : float
        Base parameter in (0, 1). Default 0.5.
    q_trainable : bool
        Whether q is trainable.

    References
    ----------
    .. [1] NIST DLMF 18.27(vii)
    .. [2] Koekoek et al. (2010), Chapter 14.28
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        **kwargs,
    ):
        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Compute Discrete q-Hermite I polynomial basis.

        Uses the three-term recurrence:

        .. math::

            h_{n+1}(x) = x \cdot h_n(x) - (1 - q^n) \cdot h_{n-1}(x)

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

        # h_0(x) = 1
        h0 = tf.ones_like(x, dtype=tf.float64)
        basis = [h0]

        if self.degree >= 1:
            # h_1(x) = x
            h1 = x
            basis.append(h1)

            # Recurrence: h_{n+1}(x) = x * h_n(x) - (1 - q^n) * h_{n-1}(x)
            h_prev = h0
            h_curr = h1

            for n in range(1, self.degree):
                q_n = tf.pow(q, tf.cast(n, tf.float64))
                C_n = 1.0 - q_n  # Coefficient of h_{n-1}

                h_next = x * h_curr - C_n * h_prev
                basis.append(h_next)
                h_prev = h_curr
                h_curr = h_next

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)


__all__ = [
    "DiscreteQHermite1",
]
