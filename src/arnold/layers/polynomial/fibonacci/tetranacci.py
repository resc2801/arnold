import tensorflow as tf
import numpy as np

from arnold.layers.polynomial.poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


class Tetranacci(PolynomialBase):
    """
    Kolmogorov-Arnold Network layer using Heptanacci polynomials.

    Tetranacci polynomials are a generalization of the Fibonacci polynomials.
    """

    @tf.function
    def poly_basis(self, x):
        """ 
        Evalute Tetranacci basis polynomials for given `x`."""

        tetranacci_basis = [ 
            tf.zeros_like(x),
            tf.ones_like(x),
            x,
            x**2,
        ]

        for n in range(4, self.degree + 1):
            tetranacci_basis.append(
                tf.math.add_n([
                    x * tetranacci_basis[n-1],
                    tetranacci_basis[n-2], 
                    tetranacci_basis[n-3],
                    tetranacci_basis[n-4]
                ])
            )

        return tf.stack(tetranacci_basis[0:(self.degree + 1)], axis=-1)
