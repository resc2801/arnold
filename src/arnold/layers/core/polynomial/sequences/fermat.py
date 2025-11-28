# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Fermat and Fermat-Lucas polynomial KAN layers.

**Fermat polynomials** :math:`F_n(x)`:

.. math::

    F_0(x) = 0, \quad F_1(x) = 1, \quad F_{n+1}(x) = 3x F_n(x) - 2 F_{n-1}(x)

**Fermat-Lucas polynomials** :math:`f_n(x)`:

.. math::

    f_0(x) = 2, \quad f_1(x) = 3x, \quad f_{n+1}(x) = 3x f_n(x) - 2 f_{n-1}(x)

Both are Lucas polynomial sequences with :math:`p(x) = 3x, q(x) = -2`.
"""
import tensorflow as tf

from arnold.utils.compilation import kan_function

from ..poly_base import PolynomialBase


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="Fermat")
class Fermat(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Fermat polynomials.

    Domain is :math:`\mathbb{R}`; growth is exponential in degree, so clipping
    inputs can help when using large orders.

    The Fermat polynomials are the w-polynomials obtained by setting
    :math:`p(x) = 3x` and :math:`q(x) = -2` in the Lucas polynomial sequence:

    .. math::

        F_0(x) = 0, \quad F_1(x) = 1

    .. math::

        F_{n+1}(x) = 3x F_n(x) - 2 F_{n-1}(x), \quad n \geq 1

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
        Vectorized Fermat polynomial evaluation.

        Recurrence: :math:`F_{n+1}(x) = 3x F_n(x) - 2 F_{n-1}(x)`.

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
        if self.degree == 0:
            return tf.expand_dims(F0, axis=-1)

        F1 = tf.ones_like(x)
        if self.degree == 1:
            return tf.stack([F0, F1], axis=-1)

        def step(carry, _):
            Fn_1, Fn_2 = carry
            Fn = 3.0 * x * Fn_1 - 2.0 * Fn_2
            return (Fn, Fn_1)

        carries = tf.scan(
            step,
            tf.range(2, self.degree + 1),
            initializer=(F1, F0),
        )
        Fs = tf.transpose(carries[0], perm=[1, 2, 0])
        fermat_basis = tf.concat(
            [tf.expand_dims(F0, -1), tf.expand_dims(F1, -1), Fs], axis=-1
        )
        return fermat_basis


@tfk.utils.register_keras_serializable(package="arnold", name="FermatLucas")
class FermatLucas(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Fermat-Lucas polynomials.

    Domain is :math:`\mathbb{R}`; behaves similarly to Lucas sequences with
    exponential growth in degree. Use ``input_clip`` for stability when inputs
    are unbounded.

    The Fermat-Lucas polynomials are the w-polynomials obtained by setting
    :math:`p(x) = 3x` and :math:`q(x) = -2` in the Lucas polynomial sequence:

    .. math::

        f_0(x) = 2, \quad f_1(x) = 3x

    .. math::

        f_{n+1}(x) = 3x f_n(x) - 2 f_{n-1}(x), \quad n \geq 1

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
        Vectorized Fermat-Lucas polynomial evaluation.

        Recurrence: :math:`f_{n+1}(x) = 3x f_n(x) - 2 f_{n-1}(x)`.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape ``(batch, input_dim)``.

        Returns
        -------
        tf.Tensor
            Basis tensor of shape ``(batch, input_dim, degree+1)``.
        """
        f0 = 2.0 * tf.ones_like(x)
        if self.degree == 0:
            return tf.expand_dims(f0, axis=-1)

        f1 = 3.0 * x
        if self.degree == 1:
            return tf.stack([f0, f1], axis=-1)

        def step(carry, _):
            fn_1, fn_2 = carry
            fn = 3.0 * x * fn_1 - 2.0 * fn_2
            return (fn, fn_1)

        carries = tf.scan(
            step,
            tf.range(2, self.degree + 1),
            initializer=(f1, f0),
        )
        fs = tf.transpose(carries[0], perm=[1, 2, 0])
        fermat_lucas_basis = tf.concat(
            [tf.expand_dims(f0, -1), tf.expand_dims(f1, -1), fs], axis=-1
        )
        return fermat_lucas_basis
