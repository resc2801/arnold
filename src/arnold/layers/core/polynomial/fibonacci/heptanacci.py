import tensorflow as tf
import numpy as np

from ..poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="Heptanacci")
class Heptanacci(PolynomialBase):
    """
    Kolmogorov-Arnold Network layer using Heptanacci polynomials.

    Heptanacci polynomials are a generalization of the Fibonacci polynomials.
    """

    @tf.function
    def poly_basis(self, x):
        heptanacci_basis = [ 
            tf.zeros_like(x),
            tf.ones_like(x),
            x,
            x,
            x**2,
            x**2,
            x**3
        ]

        for n in range(7, self.degree + 1):
            heptanacci_basis.append(
                tf.math.add_n([
                    x * heptanacci_basis[n-1],
                    heptanacci_basis[n-2], 
                    heptanacci_basis[n-3],
                    heptanacci_basis[n-4],
                    heptanacci_basis[n-5],
                    heptanacci_basis[n-6],
                    heptanacci_basis[n-7]
                ])
            )

        return tf.stack(heptanacci_basis[0:(self.degree + 1)], axis=-1)
