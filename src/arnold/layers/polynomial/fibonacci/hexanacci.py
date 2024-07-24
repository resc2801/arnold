import tensorflow as tf
import numpy as np

from arnold.layers.polynomial.poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="Hexanacci")
class Hexanacci(PolynomialBase):
    """
    Kolmogorov-Arnold Network layer using Heptanacci polynomials.

    Hexanacci polynomials are a generalization of the Fibonacci polynomials.
    """

    @tf.function
    def poly_basis(self, x):
        """ 
        Evalute Hexanacci basis polynomials for given `x`."""

        # :math:`_{0}(x) = 0`
        hexanacci_basis = [ 
            tf.zeros_like(x),
            tf.ones_like(x),
            x,
            x**2,
            x**2,
            x**3,
        ]

        for n in range(6, self.degree + 1):
            # :math:`F_{n+1}(x) = x * F_{n}(x) + F_{n-1}(x)` when n >= 1
            hexanacci_basis.append(
                tf.math.add_n([
                    x * hexanacci_basis[n-1],
                    hexanacci_basis[n-2], 
                    hexanacci_basis[n-3],
                    hexanacci_basis[n-4],
                    hexanacci_basis[n-5],
                    hexanacci_basis[n-6],
                ])
            )

        return tf.stack(hexanacci_basis[0:(self.degree + 1)], axis=-1)
