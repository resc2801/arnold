# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Tetranacci polynomial KAN layer.

The Tetranacci polynomials are defined by a 4-term recurrence:

.. math::

    T_n(x) = x T_{n-1}(x) + T_{n-2}(x) + T_{n-3}(x) + T_{n-4}(x), \quad n \geq 4
"""
import tensorflow as tf

from arnold.utils.compilation import kan_function

from ..poly_base import PolynomialBase


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="Tetranacci")
class Tetranacci(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Tetranacci polynomials.

    Tetranacci polynomials are a generalization of the Fibonacci polynomials
    using a 4-term sum recurrence.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.
    units : int
        Output dimension.
    input_clip : tuple[float, float] | None, optional
        Optional input clipping range for numerical stability.

    Notes
    -----
    The 4-term recurrence is:

    .. math::

        T_n = x T_{n-1} + T_{n-2} + T_{n-3} + T_{n-4}

    References
    ----------
    .. [1] OEIS A000078 — Tetranacci numbers
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Tetranacci polynomial evaluation.

        4-term recurrence: :math:`T_n = x T_{n-1} + T_{n-2} + T_{n-3} + T_{n-4}`.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape ``(batch, input_dim)``.

        Returns
        -------
        tf.Tensor
            Basis tensor of shape ``(batch, input_dim, degree+1)``.
        """
        T0 = tf.zeros_like(x)
        T1 = tf.ones_like(x)
        T2 = x
        T3 = x**2

        if self.degree <= 3:
            basis_list = [T0, T1, T2, T3][: (self.degree + 1)]
            return tf.stack(basis_list, axis=-1)

        def step(carry, _):
            Tn_1, Tn_2, Tn_3, Tn_4 = carry
            Tn = x * Tn_1 + Tn_2 + Tn_3 + Tn_4
            return (Tn, Tn_1, Tn_2, Tn_3)

        basis_list = [T0, T1, T2, T3]
        carry = (T3, T2, T1, T0)
        for _ in range(4, self.degree + 1):
            carry = step(carry, None)
            basis_list.append(carry[0])

        return tf.stack(basis_list, axis=-1)
