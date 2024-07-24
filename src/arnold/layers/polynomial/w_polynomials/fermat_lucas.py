import tensorflow as tf
import numpy as np

from arnold.layers.polynomial.poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="FermatLucas")
class FermatLucas(PolynomialBase):
    """
    Kolmogorov-Arnold Network layer using Fermat polynomials.

    The Fermat-Lucas polynomials are the w-polynomials obtained by setting p(x)=3*x and q(x)=-2 in the Lucas polynomial sequence:

    * :math:`w_{0}(x) = f_{0}(x) = 2`
    * :math:`w_{1}(x) = f_{1}(x) = 3*x`
    * :math:`w_{n+1}(x) = p(x) * w_{n}(x) + q(x) * w_{n-1}(x)` when n >= 1

    See also: https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    @tf.function
    def poly_basis(self, x):
        # :math:`f_{0}(x) = 2`
        fermat_lucas_basis = [ 2 * tf.ones_like(x) ]
        
        if self.degree > 0:
            # :math:`f_{1}(x) = 3 * x`
            fermat_lucas_basis.append( 3 * x)

        for n in range(2, self.degree + 1):
            # :math:`f_{n+1}(x) = 3 * x * f_{n}(x) - 2 * f_{n-1}(x)` when n >= 1
            fermat_lucas_basis.append(
                3 * x * fermat_lucas_basis[n-1] - 2 * fermat_lucas_basis[n-2]
            )
        
        return tf.stack(fermat_lucas_basis, axis=-1)
