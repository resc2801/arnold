import tensorflow as tf
import numpy as np

from arnold.layers.polynomial.poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="Pell")
class Pell(PolynomialBase):
    """
    Kolmogorov-Arnold Network layer using Pell polynomials.

    The Pell-Lucas polynomials are the w-polynomials obtained by setting p(x)=2*x and q(x)=1 in the Lucas polynomial sequence:

    * :math:`w_{0}(x) = P_{0}(x) = 0`
    * :math:`w_{1}(x) = P_{0}(x) = 1`
    * :math:`w_{n+1}(x) = p(x) * w_{n}(x) + q(x) * w_{n-1}(x)` when n >= 1

    See also: https://www.mathstat.dal.ca/FQ/Scanned/23-1/horadam.pdf
    See also: https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    @tf.function
    def poly_basis(self, x):
        # :math:`P_{0}(x) = 0`
        pell_basis = [ tf.zeros_like(x) ]
        
        if self.degree > 0:
            # :math:`P_{1}(x) = x`
            pell_basis.append( tf.ones_like(x) )

        for n in range(2, self.degree + 1):
            # :math:`P_{n+1}(x) = 2 * x * P_{n}(x) + P_{n-1}(x)` when n >= 1
            pell_basis.append(
                2 * x * pell_basis[n-1] + pell_basis[n-2]
            )
        
        return tf.stack(pell_basis, axis=-1)
