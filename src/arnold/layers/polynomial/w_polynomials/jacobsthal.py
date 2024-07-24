import tensorflow as tf
import numpy as np

from arnold.layers.polynomial.poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="Jacobsthal")
class Jacobsthal(PolynomialBase):
    """
    Kolmogorov-Arnold Network layer using Jacobsthal polynomials.

    The Jacobsthal polynomials are the w-polynomials obtained by setting p(x)=1 and q(x)=2*x in the Lucas polynomial sequence:

    * :math:`w_{0}(x) = J_{0}(x) = 0`
    * :math:`w_{1}(x) = J_{1}(x) = 1`
    * :math:`w_{n+1}(x) = p(x) * w_{n}(x) + q(x) * w_{n-1}(x)` when n >= 1

    See also: https://www.fq.math.ca/Scanned/35-2/horadam.pdf
    See also: https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    @tf.function
    def poly_basis(self, x):
        # :math:`J_{0}(x) = 2`
        jacobsthal_basis = [ tf.zeros_like(x) ]
        
        if self.degree > 0:
            # :math:`J_{1}(x) = x`
            jacobsthal_basis.append(tf.ones_like(x))

        for n in range(2, self.degree + 1):
            # :math:`J_{n+1}(x) = 1 * J_{n}(x) + 2 * x * J_{n-1}(x)` when n >= 1
            jacobsthal_basis.append(
                jacobsthal_basis[n-1] + 2 * x * jacobsthal_basis[n-2]
            )
        
        return tf.stack(jacobsthal_basis, axis=-1)
