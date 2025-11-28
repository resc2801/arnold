# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Jacobsthal and Jacobsthal-Lucas polynomial KAN layers.

**Jacobsthal polynomials** :math:`J_n(x)`:

.. math::

    J_0(x) = 0, \quad J_1(x) = 1, \quad J_{n+1}(x) = J_n(x) + 2x J_{n-1}(x)

**Jacobsthal-Lucas polynomials** :math:`j_n(x)`:

.. math::

    j_0(x) = 2, \quad j_1(x) = 1, \quad j_{n+1}(x) = j_n(x) + 2x j_{n-1}(x)

Both are Lucas polynomial sequences with :math:`p(x) = 1, q(x) = 2x`.
"""
import tensorflow as tf

from arnold.utils.compilation import kan_function
from ..poly_base import PolynomialBase


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="Jacobsthal")
class Jacobsthal(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Jacobsthal polynomials.

    Domain is :math:`\mathbb{R}`; growth is exponential in degree, so clipping
    inputs can help when using large orders.

    The Jacobsthal polynomials are the w-polynomials obtained by setting
    :math:`p(x) = 1` and :math:`q(x) = 2x` in the Lucas polynomial sequence:

    .. math::

        J_0(x) = 0, \quad J_1(x) = 1

    .. math::

        J_{n+1}(x) = J_n(x) + 2x J_{n-1}(x), \quad n \geq 1

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
    .. [1] https://www.fq.math.ca/Scanned/35-2/horadam.pdf
    .. [2] https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Jacobsthal polynomial evaluation.

        Recurrence: :math:`J_{n+1}(x) = J_n(x) + 2x J_{n-1}(x)`.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape ``(batch, input_dim)``.

        Returns
        -------
        tf.Tensor
            Basis tensor of shape ``(batch, input_dim, degree+1)``.
        """
        J0 = tf.zeros_like(x)
        if self.degree == 0:
            return tf.expand_dims(J0, axis=-1)

        J1 = tf.ones_like(x)
        if self.degree == 1:
            return tf.stack([J0, J1], axis=-1)

        def step(carry, _):
            Jn_1, Jn_2 = carry
            Jn = Jn_1 + 2.0 * x * Jn_2
            return (Jn, Jn_1)

        carries = tf.scan(
            step,
            tf.range(2, self.degree + 1),
            initializer=(J1, J0),
        )
        Js = tf.transpose(carries[0], perm=[1, 2, 0])
        jacobsthal_basis = tf.concat(
            [tf.expand_dims(J0, -1), tf.expand_dims(J1, -1), Js], axis=-1
        )
        return jacobsthal_basis


@tfk.utils.register_keras_serializable(package="arnold", name="JacobsthalLucas")
class JacobsthalLucas(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Jacobsthal-Lucas polynomials.

    Domain is :math:`\mathbb{R}`; growth is exponential in degree, so clipping
    inputs can help when using large orders.

    The Jacobsthal-Lucas polynomials are the w-polynomials obtained by setting
    :math:`p(x) = 1` and :math:`q(x) = 2x` in the Lucas polynomial sequence:

    .. math::

        j_0(x) = 2, \quad j_1(x) = 1

    .. math::

        j_{n+1}(x) = j_n(x) + 2x j_{n-1}(x), \quad n \geq 1

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
    .. [1] https://www.fq.math.ca/Scanned/35-2/horadam.pdf
    .. [2] https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Jacobsthal-Lucas polynomial evaluation.

        Recurrence: :math:`j_{n+1}(x) = j_n(x) + 2x j_{n-1}(x)`.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape ``(batch, input_dim)``.

        Returns
        -------
        tf.Tensor
            Basis tensor of shape ``(batch, input_dim, degree+1)``.
        """
        j0 = 2.0 * tf.ones_like(x)
        if self.degree == 0:
            return tf.expand_dims(j0, axis=-1)

        j1 = tf.ones_like(x)
        if self.degree == 1:
            return tf.stack([j0, j1], axis=-1)

        def step(carry, _):
            jn_1, jn_2 = carry
            jn = jn_1 + 2.0 * x * jn_2
            return (jn, jn_1)

        carries = tf.scan(
            step,
            tf.range(2, self.degree + 1),
            initializer=(j1, j0),
        )
        js = tf.transpose(carries[0], perm=[1, 2, 0])
        jacobsthal_lucas_basis = tf.concat(
            [tf.expand_dims(j0, -1), tf.expand_dims(j1, -1), js], axis=-1
        )
        return jacobsthal_lucas_basis
