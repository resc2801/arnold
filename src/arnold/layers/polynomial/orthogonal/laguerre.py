import tensorflow as tf
import numpy as np

from arnold.layers.polynomial.poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


class GeneralizedLaguerre(PolynomialBase):
    """
    Kolmogorov-Arnold Network layer using Laguerre polynomials.

    The generalized Laguerre polynomials are generated by the three-term recurrence relation:

    * :math:`L^{\alpha}_{0}(x) = 1`
    * :math:`L^{\alpha}_{1}(x) = 1 + \alpha - x`
    * :math:`L^{\alpha}_{n+1}(x) = \frac{(2n + 1 + \alpha - x) * L^{\alpha}_{n}(x) - (n + \alpha) * L^{\alpha}_{n-1}(x)}{n+1}` when n >= 1

    Special cases of the  generalized Laguerre polynomials are:
    * the Laguerre polynomials (when alpha=0); 

    See also: https://en.wikipedia.org/wiki/Laguerre_polynomials#Generalized_Laguerre_polynomials
    """

    def __init__(
            self, 
            *args,
            alpha_init: float | None = None, alpha_trainable=True, 
            **kwargs):
        
        super().__init__(*args, **kwargs)

        self.alpha_init = alpha_init
        self.alpha_trainable = alpha_trainable

        self.alpha = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.alpha_init) if self.alpha_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='alpha',
            trainable=self.alpha_trainable
        )

    @tf.function
    def poly_basis(self, x):
        """
        Evaluate generalized Laguerre basis polynomials for given `x`.
        """

        # :math:`L^{\alpha}_{0}(x) = 1`
        laguerre_basis =  [tf.ones_like(x)]
        
        if self.degree > 0:
            # :math:`L^{\alpha}_{1}(x) = 1 + \alpha - x`
            laguerre_basis.append(
                1.0 + self.alpha - x
            )

        for n in range(2, self.degree + 1):
            # :math:`L^{\alpha}_{n+1}(x) = \frac{(2n + 1 + \alpha - x) * L^{\alpha}_{n}(x) - (n + \alpha) * L^{\alpha}_{n-1}(x)}{n+1}` when n >= 1
            laguerre_basis.append(
                (((2 * n - 1 + self.alpha - x) * laguerre_basis[n-1]) - ((n - 1) * self.alpha * laguerre_basis[n-2])) / n
            )

        return tf.stack(laguerre_basis, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha_init": self.alpha_init,
            "alpha_trainable": self.alpha_trainable,
        })
        return config
