## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
import tensorflow as tf

from arnold.utils.numerics import clamp_abs

from .polynomial.poly_base import PolynomialBase


tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="Laurent")
class Laurent(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Laurent polynomials.

    Laurent polynomials may include negative powers. Inputs are internally
    clamped away from zero to avoid poles in negative-degree terms; set
    ``input_clip`` if you need tighter control of the domain.
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        input_clip=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        degree : int
            Maximum positive/negative degree.
        units : int
            Output dimensionality.
        input_clip : tuple[float, float] | None
            Optional clamp of inputs to avoid poles at 0.
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)
        self.degree = degree
        self.poly_coeffs = None

    def build(self, input_shape):
        super().build(input_shape)
        self.poly_coeffs = self.add_weight(
            shape=(self.input_dim, 2 * self.degree + 1, self.output_dim),
            initializer=tfk.initializers.RandomNormal(mean=0.0, stddev=(1.0 / (self.input_dim * (self.degree + 1)))),
            constraint=None,
            regularizer=None,
            trainable=True,
            name="polynomial_coefficients",
        )

    @tf.function(
        autograph=True,
        jit_compile=True,
        reduce_retracing=True,
        experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
    )
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Laurent basis evaluation via cumulative products.

        Evaluates :math:`\{x^{-d}, \ldots, x^{-1}, 1, x, \ldots, x^d\}` without Python loops.
        Uses separate forward/backward passes for positive/negative powers to maintain
        numerical stability.

        :param x: Input tensor, shape (batch, input_dim)
        :type x: tf.Tensor
        :returns: Laurent basis tensor, shape (batch, input_dim, 2*degree+1)
        :rtype: tf.Tensor
        """
        x_safe = clamp_abs(x, eps=1e-6)

        # Handle degree=0 case: just return 1
        if self.degree == 0:
            return tf.expand_dims(tf.ones_like(x_safe), axis=-1)

        # Positive powers: x^0, x^1, ..., x^degree via cumulative product
        def pos_step(x_acc, _):
            x_next = x_acc * x_safe
            return x_next  # Only return new accumulator

        carries_pos = tf.scan(
            pos_step,
            tf.range(self.degree),
            initializer=tf.ones_like(x_safe)
        )
        # carries_pos has shape (degree, batch, input_dim), transpose to (batch, input_dim, degree)
        pos_powers = tf.transpose(carries_pos, perm=[1, 2, 0])
        # Prepend x^0 = 1
        pos_basis = tf.concat([tf.expand_dims(tf.ones_like(x_safe), axis=-1), pos_powers], axis=-1)

        # Negative powers: x^(-1), ..., x^(-degree) via cumulative product of 1/x
        x_inv = tf.math.reciprocal(x_safe)
        def neg_step(x_acc, _):
            x_next = x_acc * x_inv
            return x_next  # Only return new accumulator

        carries_neg = tf.scan(
            neg_step,
            tf.range(self.degree),
            initializer=x_inv
        )
        # carries_neg has shape (degree, batch, input_dim), transpose to (batch, input_dim, degree)
        neg_powers = tf.transpose(carries_neg, perm=[1, 2, 0])
        # Reverse to get [x^(-degree), ..., x^(-1)]
        neg_basis = tf.reverse(neg_powers, axis=[-1])

        # Concatenate: [x^(-degree), ..., x^(-1), 1, x, ..., x^degree]
        laurent_basis = tf.concat([neg_basis, pos_basis], axis=-1)

        return laurent_basis
