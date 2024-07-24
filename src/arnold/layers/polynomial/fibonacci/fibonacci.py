import tensorflow as tf
import numpy as np

from arnold.layers.polynomial.poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers

@tfk.utils.register_keras_serializable(package="arnold", name="Fibonacci")
class Fibonacci(PolynomialBase):
    """
    Kolmogorov-Arnold Network layer using Fibonacci polynomials.

    The Fibonacci polynomials are the w-polynomials obtained by setting p(x)=x and q(x)=1 in the Lucas polynomial sequence:

    * :math:`w_{0}(x) = F_{0}(x) = 0`
    * :math:`w_{1}(x) = F_{1}(x) = 1`
    * :math:`w_{n+1}(x) = x * w_{n}(x) + w_{n-1}(x)` when n >= 1

    See also: https://en.wikipedia.org/wiki/Fibonacci_polynomials#Definition
    See also: https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    @tf.function
    def poly_basis(self, x):
        """ 
        Evalute Fibbonaci basis polynomials for given `x`."""

        # :math:`F_{0}(x) = 0`
        # :math:`F_{1}(x) = 1`
        fibonacci_basis = [ 
            tf.zeros_like(x),
            tf.ones_like(x)
        ]

        for n in range(2, self.degree + 1):
            # :math:`F_{n+1}(x) = x * F_{n}(x) + F_{n-1}(x)` when n >= 1
            fibonacci_basis.append(
                tf.math.add_n(
                    x * fibonacci_basis[n-1],
                    fibonacci_basis[n-2]
                )
            )

        return tf.stack(fibonacci_basis[0:(self.degree + 1)], axis=-1)
