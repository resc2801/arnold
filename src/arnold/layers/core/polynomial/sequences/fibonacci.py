# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Fibonacci polynomial KAN layer.

The Fibonacci polynomials :math:`F_n(x)` are defined by the recurrence:

.. math::

    F_0(x) = 0, \quad F_1(x) = 1, \quad F_{n+1}(x) = x F_n(x) + F_{n-1}(x)

This is the w-polynomial sequence with :math:`p(x) = x, q(x) = 1`.
"""
import tensorflow as tf

from arnold.utils.compilation import kan_function
from ..poly_base import PolynomialBase


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="Fibonacci")
class Fibonacci(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Fibonacci polynomials.

    The Fibonacci polynomials are the w-polynomials obtained by setting
    :math:`p(x) = x` and :math:`q(x) = 1` in the Lucas polynomial sequence:

    .. math::

        F_0(x) = 0, \quad F_1(x) = 1

    .. math::

        F_{n+1}(x) = x F_n(x) + F_{n-1}(x), \quad n \geq 1

    The first few Fibonacci polynomials are:

    - :math:`F_0 = 0`
    - :math:`F_1 = 1`
    - :math:`F_2 = x`
    - :math:`F_3 = x^2 + 1`
    - :math:`F_4 = x^3 + 2x`
    - :math:`F_5 = x^4 + 3x^2 + 1`

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
    - Fibonacci numbers are :math:`F_n(1) = 0, 1, 1, 2, 3, 5, 8, 13, \ldots`
    - Growth is polynomial in :math:`x` with leading term :math:`x^{n-1}`

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Fibonacci_polynomials
    .. [2] https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Fibonacci polynomial evaluation.

        Recurrence: :math:`F_{n+1}(x) = x F_n(x) + F_{n-1}(x)`.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape ``(batch, input_dim)``.

        Returns
        -------
        tf.Tensor
            Basis tensor of shape ``(batch, input_dim, degree+1)``.
        """
        F0 = tf.zeros_like(x)
        F1 = tf.ones_like(x)

        if self.degree == 0:
            return tf.expand_dims(F0, axis=-1)
        if self.degree == 1:
            return tf.stack([F0, F1], axis=-1)

        def step(carry, _):
            Fn_1, Fn_2 = carry
            Fn = x * Fn_1 + Fn_2
            return (Fn, Fn_1)

        basis_list = [F0, F1]
        Fn_1, Fn_2 = F1, F0
        for _ in range(2, self.degree + 1):
            Fn_1, Fn_2 = step((Fn_1, Fn_2), None)
            basis_list.append(Fn_1)

        return tf.stack(basis_list, axis=-1)
