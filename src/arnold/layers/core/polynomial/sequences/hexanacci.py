# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Hexanacci polynomial KAN layer.

The Hexanacci polynomials are defined by a 6-term recurrence:

.. math::

    H_n(x) = x H_{n-1}(x) + \sum_{i=n-5}^{n-2} H_i(x), \quad n \geq 6
"""
import tensorflow as tf

from arnold.utils.compilation import kan_function

from ..poly_base import PolynomialBase


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="Hexanacci")
class Hexanacci(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Hexanacci polynomials.

    Hexanacci polynomials are a generalization of the Fibonacci polynomials
    using a 6-term sum recurrence.

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
    The 6-term recurrence is:

    .. math::

        H_n = x H_{n-1} + H_{n-2} + H_{n-3} + H_{n-4} + H_{n-5} + H_{n-6}

    References
    ----------
    .. [1] OEIS A001592 — Hexanacci numbers
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Hexanacci polynomial evaluation.

        6-term recurrence: :math:`H_n = x H_{n-1} + \sum_{i=n-5}^{n-2} H_i`.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape ``(batch, input_dim)``.

        Returns
        -------
        tf.Tensor
            Basis tensor of shape ``(batch, input_dim, degree+1)``.
        """
        H0 = tf.zeros_like(x)
        H1 = tf.ones_like(x)
        H2 = x
        H3 = x**2
        H4 = x**2
        H5 = x**3

        if self.degree <= 5:
            basis_list = [H0, H1, H2, H3, H4, H5][: (self.degree + 1)]
            return tf.stack(basis_list, axis=-1)

        def step(carry, _):
            Hn_1, Hn_2, Hn_3, Hn_4, Hn_5, Hn_6 = carry
            Hn = x * Hn_1 + Hn_2 + Hn_3 + Hn_4 + Hn_5 + Hn_6
            return (Hn, Hn_1, Hn_2, Hn_3, Hn_4, Hn_5)

        basis_list = [H0, H1, H2, H3, H4, H5]
        carry = (H5, H4, H3, H2, H1, H0)
        for _ in range(6, self.degree + 1):
            carry = step(carry, None)
            basis_list.append(carry[0])

        return tf.stack(basis_list, axis=-1)
