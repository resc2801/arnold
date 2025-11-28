# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Discrete q-Hermite II polynomials.

This module implements Discrete q-Hermite II polynomials :math:`\tilde{h}_n(x; q)`.

Mathematical Background
-----------------------
The Discrete q-Hermite II polynomials are defined as:

.. math::

    \tilde{h}_n(x; q) = x^n \cdot {}_2\phi_1\left(\begin{array}{c}
        q^{-n}, q^{-n+1} \\ 0 \end{array}; q^2, -\frac{q^2}{x^2}\right)

They satisfy the three-term recurrence:

.. math::

    x \, \tilde{h}_n(x; q) = \tilde{h}_{n+1}(x; q) + q^{n-1}(1 - q^n) \, \tilde{h}_{n-1}(x; q)

with initial conditions :math:`\tilde{h}_0(x; q) = 1` and :math:`\tilde{h}_1(x; q) = x`.

Properties
----------
- Symmetric: :math:`\tilde{h}_n(-x; q) = (-1)^n \tilde{h}_n(x; q)`
- The measure is indeterminate (not uniquely determined)
- Same q → 1 limit as Discrete q-Hermite I

References
----------
.. [1] NIST DLMF 18.27(vii)
.. [2] Koekoek et al. (2010), Chapter 14.29
"""

import tensorflow as tf

from arnold.utils.compilation import kan_function

from .base import QPolynomialBase


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="DiscreteQHermite2")
class DiscreteQHermite2(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Discrete q-Hermite II polynomials.

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
    .. [2] Koekoek et al. (2010), Chapter 14.29
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
        Compute Discrete q-Hermite II polynomial basis.

        Uses the three-term recurrence:

        .. math::

            \tilde{h}_{n+1}(x) = x \cdot \tilde{h}_n(x) - q^{n-1}(1 - q^n) \cdot \tilde{h}_{n-1}(x)

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

        # h̃_0(x) = 1
        h0 = tf.ones_like(x, dtype=tf.float64)
        basis = [h0]

        if self.degree >= 1:
            # h̃_1(x) = x
            h1 = x
            basis.append(h1)

            # Recurrence: h̃_{n+1}(x) = x * h̃_n(x) - q^{n-1}(1 - q^n) * h̃_{n-1}(x)
            h_prev = h0
            h_curr = h1

            for n in range(1, self.degree):
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)
                q_nm1 = tf.pow(q, n_f - 1.0)
                C_n = q_nm1 * (1.0 - q_n)  # Coefficient of h̃_{n-1}

                h_next = x * h_curr - C_n * h_prev
                basis.append(h_next)
                h_prev = h_curr
                h_curr = h_next

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)


__all__ = [
    "DiscreteQHermite2",
]
