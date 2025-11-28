# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Tribonacci polynomial KAN layer.

The Tribonacci polynomials :math:`T_n(x)` are defined by the 3-term recurrence:

.. math::

    T_0(x) = 0, \quad T_1(x) = 1, \quad T_2(x) = x

.. math::

    T_n(x) = x T_{n-1}(x) + T_{n-2}(x) + T_{n-3}(x), \quad n \geq 3
"""
import tensorflow as tf

from arnold.utils.compilation import kan_function

from ..poly_base import PolynomialBase


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="Tribonacci")
class Tribonacci(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Tribonacci polynomials.

    The Tribonacci polynomials :math:`T_n(x)` are a generalization of the
    Fibonacci polynomials using a 3-term sum recurrence:

    .. math::

        T_0(x) = 0, \quad T_1(x) = 1, \quad T_2(x) = x

    .. math::

        T_n(x) = x T_{n-1}(x) + T_{n-2}(x) + T_{n-3}(x), \quad n \geq 3

    The first few Tribonacci polynomials are:

    - :math:`T_0 = 0`
    - :math:`T_1 = 1`
    - :math:`T_2 = x`
    - :math:`T_3 = x^2 + 1`
    - :math:`T_4 = x^3 + 2x`
    - :math:`T_5 = x^4 + 3x^2 + 1`

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
    - Tribonacci numbers are :math:`T_n(1) = 0, 1, 1, 2, 4, 7, 13, 24, \ldots`
    - The characteristic equation is :math:`t^3 = t^2 + t + 1`
    - Related to ternary representations and combinatorics

    References
    ----------
    .. [1] OEIS A000073 — Tribonacci numbers
    .. [2] Spickerman, W.R. (1982). "Binet's formula for the Tribonacci sequence"
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Tribonacci polynomial evaluation.

        3-term recurrence: :math:`T_n = x T_{n-1} + T_{n-2} + T_{n-3}`.

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

        if self.degree <= 2:
            basis_list = [T0, T1, T2][: (self.degree + 1)]
            return tf.stack(basis_list, axis=-1)

        def step(carry, _):
            Tn_1, Tn_2, Tn_3 = carry
            Tn = x * Tn_1 + Tn_2 + Tn_3
            return (Tn, Tn_1, Tn_2)

        basis_list = [T0, T1, T2]
        carry = (T2, T1, T0)
        for _ in range(3, self.degree + 1):
            carry = step(carry, None)
            basis_list.append(carry[0])

        return tf.stack(basis_list, axis=-1)
