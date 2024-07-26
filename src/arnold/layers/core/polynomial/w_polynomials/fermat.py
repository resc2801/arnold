import tensorflow as tf
import numpy as np

from ..poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="Fermat")
class Fermat(PolynomialBase):
    """
    Kolmogorov-Arnold Network layer using Fermat polynomials.

    The Fermat polynomials are the w-polynomials obtained by setting p(x)=3*x and q(x)=-2 in the Lucas polynomial sequence:

    * :math:`w_{0}(x) = F_{0}(x) = 0`
    * :math:`w_{1}(x) = F_{1}(x) = 1`
    * :math:`w_{n+1}(x) = p(x) * w_{n}(x) + q(x) * w_{n-1}(x)` when n >= 1

    See also: https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    @tf.function
    def poly_basis(self, x):
        # :math:`F_{0}(x) = 0`
        fermat_basis = [ tf.zeros_like(x) ]
        
        if self.degree > 0:
            # :math:`F_{1}(x) = 1`
            fermat_basis.append(tf.ones_like(x))

        for n in range(2, self.degree + 1):
            # :math:`F_{n+1}(x) = 3 * x * F_{n}(x) - 2 * F_{n-1}(x)` when n >= 1
            fermat_basis.append(
                3 * x * fermat_basis[n-1] - 2 * fermat_basis[n-2]
            )
        
        return tf.stack(fermat_basis, axis=-1)
