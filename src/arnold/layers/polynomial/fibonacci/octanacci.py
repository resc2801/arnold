import tensorflow as tf
import numpy as np

from arnold.layers.polynomial.poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


class Octanacci(PolynomialBase):
    """
    Kolmogorov-Arnold Network layer using Heptanacci polynomials.

    Octanacci polynomials are a generalization of the Fibonacci polynomials.
    """

    @tf.function
    def poly_basis(self, x):
        """ 
        Evalute Octanacci basis polynomials for given `x`."""

        octanacci_basis = [ 
            tf.zeros_like(x),
            tf.ones_like(x),
            x,
            x**2,
            x**2,
            x**3,
            x**3,
            x**4
        ]

        for n in range(8, self.degree + 1):
            octanacci_basis.append(
                tf.math.add_n([
                    x * octanacci_basis[n-1],
                    octanacci_basis[n-2], 
                    octanacci_basis[n-3],
                    octanacci_basis[n-4],
                    octanacci_basis[n-5],
                    octanacci_basis[n-6],
                    octanacci_basis[n-7],
                    octanacci_basis[n-8]
                ])
            )

        return tf.stack(octanacci_basis[0:(self.degree + 1)], axis=-1)
