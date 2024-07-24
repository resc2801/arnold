import tensorflow as tf
import numpy as np

from arnold.layers.polynomial.poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="PellLucas")
class PellLucas(PolynomialBase):
    """
    Kolmogorov-Arnold Network layer using Pell-Lucas polynomials.

    The Pell-Lucas polynomials are the w-polynomials obtained by setting p(x)=2*x and q(x)=1 in the Lucas polynomial sequence:

    * :math:`w_{0}(x) = Q_{0}(x) = 2`
    * :math:`w_{1}(x) = Q_{0}(x) = 2x`
    * :math:`w_{n+1}(x) = p(x) * w_{n}(x) + q(x) * w_{n-1}(x)` when n >= 1

    See also: https://www.mathstat.dal.ca/FQ/Scanned/23-1/horadam.pdf
    See also: https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    @tf.function
    def poly_basis(self, x):
        # :math:`Q_{0}(x) = 2`
        pell_lucas_basis = [ 2 * tf.ones_like(x) ]
        
        if self.degree > 0:
            # :math:`Q_{1}(x) = x`
            pell_lucas_basis.append(2 * x)

        for n in range(2, self.degree + 1):
            # :math:`Q_{n+1}(x) = x * Q_{n}(x) + Q_{n-1}(x)` when n >= 1
            pell_lucas_basis.append(
                2 * x * pell_lucas_basis[n-1] + pell_lucas_basis[n-2]
            )
        
        return tf.stack(pell_lucas_basis, axis=-1)
