# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Pell and Pell-Lucas polynomial KAN layers.

**Pell polynomials** :math:`P_n(x)`:

.. math::

    P_0(x) = 0, \quad P_1(x) = 1, \quad P_{n+1}(x) = 2x P_n(x) + P_{n-1}(x)

**Pell-Lucas polynomials** :math:`Q_n(x)`:

.. math::

    Q_0(x) = 2, \quad Q_1(x) = 2x, \quad Q_{n+1}(x) = 2x Q_n(x) + Q_{n-1}(x)

Both are Lucas polynomial sequences with :math:`p(x) = 2x, q(x) = 1`.
"""
import tensorflow as tf

from arnold.utils.compilation import kan_function

from ..poly_base import PolynomialBase


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="Pell")
class Pell(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Pell polynomials.

    The Pell polynomials are the w-polynomials obtained by setting
    :math:`p(x) = 2x` and :math:`q(x) = 1` in the Lucas polynomial sequence:

    .. math::

        P_0(x) = 0, \quad P_1(x) = 1

    .. math::

        P_{n+1}(x) = 2x P_n(x) + P_{n-1}(x), \quad n \geq 1

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.
    units : int
        Output dimension.
    input_clip : tuple[float, float] | None, optional
        Optional clamp of inputs before basis evaluation.

    References
    ----------
    .. [1] https://www.mathstat.dal.ca/FQ/Scanned/23-1/horadam.pdf
    .. [2] https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Pell polynomial evaluation.

        Recurrence: :math:`P_{n+1}(x) = 2x P_n(x) + P_{n-1}(x)`.

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
        if self.degree == 0:
            return tf.expand_dims(P0, axis=-1)

        P1 = tf.ones_like(x)
        if self.degree == 1:
            return tf.stack([P0, P1], axis=-1)

        def step(carry, _):
            Pn_1, Pn_2 = carry
            Pn = 2.0 * x * Pn_1 + Pn_2
            return (Pn, Pn_1)

        carries = tf.scan(
            step,
            tf.range(2, self.degree + 1),
            initializer=(P1, P0),
        )
        Ps = tf.transpose(carries[0], perm=[1, 2, 0])
        pell_basis = tf.concat(
            [tf.expand_dims(P0, -1), tf.expand_dims(P1, -1), Ps], axis=-1
        )
        return pell_basis


@tfk.utils.register_keras_serializable(package="arnold", name="PellLucas")
class PellLucas(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Pell-Lucas polynomials.

    Domain is :math:`\mathbb{R}`; growth is exponential in degree, so clipping
    inputs can help when using large orders.

    The Pell-Lucas polynomials are the w-polynomials obtained by setting
    :math:`p(x) = 2x` and :math:`q(x) = 1` in the Lucas polynomial sequence:

    .. math::

        Q_0(x) = 2, \quad Q_1(x) = 2x

    .. math::

        Q_{n+1}(x) = 2x Q_n(x) + Q_{n-1}(x), \quad n \geq 1

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.
    units : int
        Output dimension.
    input_clip : tuple[float, float] | None, optional
        Optional clamp of inputs before basis evaluation.

    References
    ----------
    .. [1] https://www.mathstat.dal.ca/FQ/Scanned/23-1/horadam.pdf
    .. [2] https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Pell-Lucas polynomial evaluation.

        Recurrence: :math:`Q_{n+1}(x) = 2x Q_n(x) + Q_{n-1}(x)`.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape ``(batch, input_dim)``.

        Returns
        -------
        tf.Tensor
            Basis tensor of shape ``(batch, input_dim, degree+1)``.
        """
        Q0 = 2.0 * tf.ones_like(x)
        if self.degree == 0:
            return tf.expand_dims(Q0, axis=-1)

        Q1 = 2.0 * x
        if self.degree == 1:
            return tf.stack([Q0, Q1], axis=-1)

        def step(carry, _):
            Qn_1, Qn_2 = carry
            Qn = 2.0 * x * Qn_1 + Qn_2
            return (Qn, Qn_1)

        carries = tf.scan(
            step,
            tf.range(2, self.degree + 1),
            initializer=(Q1, Q0),
        )
        Qs = tf.transpose(carries[0], perm=[1, 2, 0])
        pell_lucas_basis = tf.concat(
            [tf.expand_dims(Q0, -1), tf.expand_dims(Q1, -1), Qs], axis=-1
        )
        return pell_lucas_basis
