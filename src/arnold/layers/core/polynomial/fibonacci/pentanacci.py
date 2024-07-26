import tensorflow as tf
import numpy as np

from ..poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="Pentanacci")
class Pentanacci(PolynomialBase):
    """
    Kolmogorov-Arnold Network layer using Heptanacci polynomials.

    Pentanacci polynomials are a generalization of the Fibonacci polynomials.
    """

    @tf.function
    def poly_basis(self, x):
        pentanacci_basis = [ 
            tf.zeros_like(x),
            tf.ones_like(x),
            x,
            x,
            x**2,
        ]

        for n in range(5, self.degree + 1):
            pentanacci_basis.append(
                tf.math.add_n([
                    x * pentanacci_basis[n-1],
                    pentanacci_basis[n-2], 
                    pentanacci_basis[n-3],
                    pentanacci_basis[n-4],
                    pentanacci_basis[n-5],
                ])
            )

        return tf.stack(pentanacci_basis[0:(self.degree + 1)], axis=-1)
