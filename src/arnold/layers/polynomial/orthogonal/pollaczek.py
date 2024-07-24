import tensorflow as tf
import numpy as np

from arnold.layers.polynomial.poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="Pollaczek")
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

        self.a_init = a_init
        self.a_trainable = a_trainable
        self.b_init = b_init
        self.b_trainable = b_trainable

        self.a = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.a_init) if self.a_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='a',
            trainable=self.a_trainable
        )

        self.b = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.b_init) if self.b_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='b',
            trainable=self.b_trainable
        )

    @tf.function
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "a_init": self.a_init,
            "a_trainable": self.a_trainable,
            "b_init": self.b_init,
            "b_trainable": self.b_trainable,
        })
        return config
