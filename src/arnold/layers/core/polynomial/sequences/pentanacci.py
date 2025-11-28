# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Pentanacci polynomial KAN layer.

The Pentanacci polynomials are defined by a 5-term recurrence:

.. math::

    P_n(x) = x P_{n-1}(x) + P_{n-2}(x) + P_{n-3}(x) + P_{n-4}(x) + P_{n-5}(x)
"""
import tensorflow as tf

from arnold.utils.compilation import kan_function
from ..poly_base import PolynomialBase


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="Pentanacci")
class Pentanacci(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Pentanacci polynomials.

    Pentanacci polynomials are a generalization of the Fibonacci polynomials
    using a 5-term sum recurrence.

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
    The 5-term recurrence is:

    .. math::

        P_n = x P_{n-1} + P_{n-2} + P_{n-3} + P_{n-4} + P_{n-5}

    References
    ----------
    .. [1] OEIS A001591 — Pentanacci numbers
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Pentanacci polynomial evaluation.

        5-term recurrence: :math:`P_n = x P_{n-1} + \sum_{i=n-4}^{n-2} P_i`.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape ``(batch, input_dim)``.

        Returns
        -------
        tf.Tensor
            Basis tensor of shape ``(batch, input_dim, degree+1)``.
        """
        P0 = tf.zeros_like(x)
        P1 = tf.ones_like(x)
        P2 = x
        P3 = x
        P4 = x**2

        if self.degree <= 4:
            basis_list = [P0, P1, P2, P3, P4][: (self.degree + 1)]
            return tf.stack(basis_list, axis=-1)

        def step(carry, _):
            Pn_1, Pn_2, Pn_3, Pn_4, Pn_5 = carry
            Pn = x * Pn_1 + Pn_2 + Pn_3 + Pn_4 + Pn_5
            return (Pn, Pn_1, Pn_2, Pn_3, Pn_4)

        basis_list = [P0, P1, P2, P3, P4]
        carry = (P4, P3, P2, P1, P0)
        for _ in range(5, self.degree + 1):
            carry = step(carry, None)
            basis_list.append(carry[0])

        return tf.stack(basis_list, axis=-1)
