# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Lucas polynomial KAN layer.

The Lucas polynomials :math:`L_n(x)` are defined by the recurrence:

.. math::

    L_0(x) = 2, \quad L_1(x) = x, \quad L_{n+1}(x) = x L_n(x) + L_{n-1}(x)

This is the w-polynomial sequence with :math:`p(x) = x, q(x) = 1, (a, b) = (2, x)`.
"""
import tensorflow as tf

from arnold.utils.compilation import kan_function

from ..poly_base import PolynomialBase


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="Lucas")
class Lucas(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Lucas polynomials.

    Domain is :math:`\mathbb{R}`; growth is exponential in degree, so clipping
    inputs can help when using large orders.

    The Lucas polynomials are the w-polynomials obtained by setting
    :math:`p(x) = x` and :math:`q(x) = 1` in the Lucas polynomial sequence:

    .. math::

        L_0(x) = 2, \quad L_1(x) = x

    .. math::

        L_{n+1}(x) = x L_n(x) + L_{n-1}(x), \quad n \geq 1

    It is given explicitly by:

    .. math::

        L_n(x) = 2^{-n} \left[ \left(x - \sqrt{x^2 + 4}\right)^n +
                 \left(x + \sqrt{x^2 + 4}\right)^n \right]

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
    .. [1] https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Lucas polynomial evaluation.

        Recurrence: :math:`L_{n+1}(x) = x L_n(x) + L_{n-1}(x)`.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape ``(batch, input_dim)``.

        Returns
        -------
        tf.Tensor
            Basis tensor of shape ``(batch, input_dim, degree+1)``.
        """
        L0 = 2.0 * tf.ones_like(x)
        if self.degree == 0:
            return tf.expand_dims(L0, axis=-1)

        L1 = x
        if self.degree == 1:
            return tf.stack([L0, L1], axis=-1)

        def step(carry, _):
            Ln_1, Ln_2 = carry
            Ln = x * Ln_1 + Ln_2
            return (Ln, Ln_1)

        carries = tf.scan(
            step,
            tf.range(2, self.degree + 1),
            initializer=(L1, L0),
        )
        # carries[0] has shape (num_steps, batch, input_dim)
        Ls = tf.transpose(carries[0], perm=[1, 2, 0])
        lucas_basis = tf.concat(
            [tf.expand_dims(L0, -1), tf.expand_dims(L1, -1), Ls], axis=-1
        )
        return lucas_basis
