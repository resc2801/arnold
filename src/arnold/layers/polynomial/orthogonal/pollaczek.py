import tensorflow as tf
import numpy as np

from arnold.layers.polynomial.poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


class Pollaczek(PolynomialBase):
    """
    Kolmogorov-Arnold Network layer using Pollaczek polynomials.

    The Pollaczek polynomials are generated by the three-term recurrence relation:

    * :math:`P_{0}(x; a, b) = 1`
    * :math:`P_{1}(x; a, b) = (2 * a + 1) * x + 2 * b`
    * :math:`P_{n}(x;a,b) = \frac{[(2n-1+2a)x+2b]P_{n-1}(x;a,b)-(n-1)P_{n-2}(x;a,b) }{n} when n >= 2

    See also: https://mathworld.wolfram.com/PollaczekPolynomial.html
    """

    def __init__(
            self, 
            *args,
            a_init: float | None = None, a_trainable=True, 
            b_init: float | None = None, b_trainable=True,
            **kwargs):
        
        super().__init__(*args, **kwargs)

        self.a = self.add_weight(
            initializer=tfk.initializers.Constant(value=a_init) if a_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='a',
            trainable=a_trainable
        )

        self.b = self.add_weight(
            initializer=tfk.initializers.Constant(value=b_init) if b_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='b',
            trainable=b_trainable
        )

    def poly_basis(self, x):
        """
        Evaluate Pollaczek basis polynomials for given `x`.
        """

        # :math:`P_{0}(x; a, b) = 1`
        pollaczek_basis = [ tf.ones_like(x) ]
        
        if self.degree > 0:
            # :math:`P_{1}(x; a, b) = (2 * a + 1) * x + 2 * b`
            pollaczek_basis.append(
                (2 * self.a + 1) * x + 2 * self.b
            )

        for n in range(2, self.degree + 1):
            # :math:`P_{n}(x;a,b) = \frac{[(2n-1+2a)x+2b]P_{n-1}(x;a,b)-(n-1)P_{n-2}(x;a,b) }{n} when n >= 2
            pollaczek_basis.append(
                (((2 * n - 1 + 2 * self.a) * x + 2* self.b) * pollaczek_basis[n-1] - (n-1) * pollaczek_basis[n-2]) / n
            )

        return tf.stack(pollaczek_basis, axis=-1)
