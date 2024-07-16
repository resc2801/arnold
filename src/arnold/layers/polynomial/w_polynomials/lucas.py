import tensorflow as tf
import numpy as np

from arnold.layers.polynomial.poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


class Lucas(PolynomialBase):
    """
    Kolmogorov-Arnold Network layer using Lucas polynomials.

    The Lucas polynomials are the w-polynomials obtained by setting p(x)=x and q(x)=1 in the Lucas polynomial sequence:

    * :math:`w_{0}(x) = L_{0}(x) = 2`
    * :math:`w_{1}(x) = L_{0}(x) = x`
    * :math:`w_{n+1}(x) = p(x) * w_{n}(x) + q(x) * w_{n-1}(x)` when n >= 1


    It is given explicitly by

    :math:`L_{n}(x) = 2^{-n} * ( (x - \sqrt(x^{2} + 4) )^{n} + (x + \sqrt(x^{2} + 4) )^{n} )`

    See also: https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    @tf.function
    def poly_basis(self, x):
        # :math:`L_{0}(x) = 2`
        lucas_basis = [ 2 * tf.ones_like(x) ]
        
        if self.degree > 0:
            # :math:`L_{1}(x) = x`
            lucas_basis.append(x)

        for n in range(2, self.degree + 1):
            # :math:`L_{n+1}(x) = x * L_{n}(x) + L_{n-1}(x)` when n >= 1
            lucas_basis.append(
                x * lucas_basis[n-1] + lucas_basis[n-2]
            )
        
        return tf.stack(lucas_basis, axis=-1)
