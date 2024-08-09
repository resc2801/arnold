import tensorflow as tf
import numpy as np

from .poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="Boubaker")
class Boubaker(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using the Boubaker polynomials.

    See: https://en.wikiversity.org/wiki/Boubaker_Polynomials
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        boubaker_basis = [ tf.ones_like(x) ]

        if self.degree > 0:
            boubaker_basis.append(x)
        
        if self.degree > 1:
            boubaker_basis.append(tf.square(x) + 2)

        for n in range(3, self.degree + 1):
            boubaker_basis.append(
                x * boubaker_basis[n-1] - boubaker_basis[n-2]
            )

        return tf.stack(boubaker_basis, axis=-1) 

