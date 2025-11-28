# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Octanacci polynomial KAN layer.

The Octanacci polynomials are defined by an 8-term recurrence:

.. math::

    O_n(x) = x O_{n-1}(x) + \sum_{i=n-7}^{n-2} O_i(x), \quad n \geq 8
"""
import tensorflow as tf

from arnold.utils.compilation import kan_function
from ..poly_base import PolynomialBase


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="Octanacci")
class Octanacci(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Octanacci polynomials.

    Octanacci polynomials are a generalization of the Fibonacci polynomials
    using an 8-term sum recurrence.

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
    The 8-term recurrence is:

    .. math::

        O_n = x O_{n-1} + O_{n-2} + O_{n-3} + O_{n-4} + O_{n-5} + O_{n-6} + O_{n-7} + O_{n-8}

    References
    ----------
    .. [1] OEIS A079262 — Octanacci numbers
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Octanacci polynomial evaluation.

        8-term recurrence: :math:`O_n = x O_{n-1} + \sum_{i=n-7}^{n-2} O_i`.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape ``(batch, input_dim)``.

        Returns
        -------
        tf.Tensor
            Basis tensor of shape ``(batch, input_dim, degree+1)``.
        """
        O0 = tf.zeros_like(x)
        O1 = tf.ones_like(x)
        O2 = x
        O3 = x**2
        O4 = x**2
        O5 = x**3
        O6 = x**3
        O7 = x**4

        if self.degree <= 7:
            basis_list = [O0, O1, O2, O3, O4, O5, O6, O7][: (self.degree + 1)]
            return tf.stack(basis_list, axis=-1)

        def step(carry, _):
            On_1, On_2, On_3, On_4, On_5, On_6, On_7, On_8 = carry
            On = x * On_1 + On_2 + On_3 + On_4 + On_5 + On_6 + On_7 + On_8
            return (On, On_1, On_2, On_3, On_4, On_5, On_6, On_7)

        basis_list = [O0, O1, O2, O3, O4, O5, O6, O7]
        carry = (O7, O6, O5, O4, O3, O2, O1, O0)
        for _ in range(8, self.degree + 1):
            carry = step(carry, None)
            basis_list.append(carry[0])

        return tf.stack(basis_list, axis=-1)
