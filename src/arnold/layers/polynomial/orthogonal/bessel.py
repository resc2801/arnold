import tensorflow as tf
import numpy as np

from arnold.layers.polynomial.poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


class Bessel(PolynomialBase):
    """
    Kolmogorov-Arnold Network layer using Bessel polynomials.

    The Bessel polynomials are generated by the three-term recurrence relation:

    * :math:`y_{0}(x) = 1`
    * :math:`y_{1}(x) = x + 1`
    * :math:`y_{n}(x) = (2n - 1) * x * y_{n-1}(x) + y_{n-2}(x)` when n >= 2
    
    See also: https://en.wikipedia.org/wiki/Bessel_polynomials#Recursion
    """

    @tf.function
    def poly_basis(self, x):
        """
        Evaluate Bessel basis polynomials for `x`.
        """
        # :math:`y_{0}(x) = 1`
        bessel_basis = [ tf.ones_like(x) ]

        if self.degree > 0:
            # :math:`y_{1}(x) = x + 1`
            bessel_basis.append(x + 1.0)

        for n in range(2, self.degree + 1):
            # :math:`y_{n}(x) = (2n - 1) * x * y_{n-1}(x) + y_{n-2}(x)` when n >= 2
            bessel_basis.append(
                (2 * n - 1) * x * bessel_basis[n-1] + bessel_basis[n-2]
            )

        return tf.stack(bessel_basis, axis=-1)
