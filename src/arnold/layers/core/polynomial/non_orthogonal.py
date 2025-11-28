## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
import tensorflow as tf

from .poly_base import PolynomialBase


tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="Boubaker")
class Boubaker(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using the Boubaker polynomials.

    Domain is :math:`\mathbb{R}`; growth is moderate compared to Hermite/Bessel,
    but large degrees can still amplify inputs without clipping. Use
    ``input_clip`` for stability on unbounded inputs.

    See: https://en.wikiversity.org/wiki/Boubaker_Polynomials
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        """
        Parameters
        ----------
        degree : int
            Maximum polynomial degree.
        units : int
            Output dimensionality.
        input_clip : tuple[float, float] | None
            Optional clamp of inputs before basis evaluation.
        **kwargs :
            Forwarded to :class:`PolynomialBase` (e.g., activation, use_bias).
        """
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @tf.function(
        autograph=True,
        jit_compile=True,
        reduce_retracing=True,
        experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
    )
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Boubaker polynomial evaluation via tf.scan.
        
        Recurrence: :math:`B_n(x) = x B_{n-1}(x) - B_{n-2}(x)` for :math:`n \geq 3`.
        """
        B0 = tf.ones_like(x)
        if self.degree == 0:
            return tf.expand_dims(B0, axis=-1)
        
        B1 = x
        if self.degree == 1:
            return tf.stack([B0, B1], axis=-1)
        
        B2 = tf.square(x) + 2.0
        if self.degree == 2:
            return tf.stack([B0, B1, B2], axis=-1)
        
        # Use tf.scan for n >= 3
        def step(carry, _):
            Bn_1, Bn_2 = carry
            Bn = x * Bn_1 - Bn_2
            return (Bn, Bn_1)  # Only return new carry
        
        carries = tf.scan(
            step,
            tf.range(3, self.degree + 1),
            initializer=(B2, B1)
        )
        # carries[0] has shape (num_steps, batch, input_dim), transpose to (batch, input_dim, num_steps)
        Bs = tf.transpose(carries[0], perm=[1, 2, 0])
        boubaker_basis = tf.concat([tf.expand_dims(B0, -1), tf.expand_dims(B1, -1), tf.expand_dims(B2, -1), Bs], axis=-1)
        return boubaker_basis
