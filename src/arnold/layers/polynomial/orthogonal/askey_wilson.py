import tensorflow as tf
import numpy as np

from arnold.layers.polynomial.poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


class AskeyWilson(PolynomialBase):
    """
    Kolmogorov-Arnold Network layer using the Askey-Wilson polynomials.
    """

    def __init__(
            self, 
            *args,
            a_init: float | None = None, a_trainable=True, 
            b_init: float | None = None, b_trainable=True,
            c_init: float | None = None, c_trainable=True, 
            d_init: float | None = None, d_trainable=True, 
            q_init: float | None = None, q_trainable=True,
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

        self.c = self.add_weight(
            initializer=tfk.initializers.Constant(value=c_init) if c_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='c',
            trainable=c_trainable
        )

        self.d = self.add_weight(
            initializer=tfk.initializers.Constant(value=d_init) if d_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='d',
            trainable=d_trainable
        )

        self.q = self.add_weight(
            initializer=tfk.initializers.Constant(value=q_init) if q_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='q',
            trainable=q_trainable
        )

    @tf.function
    def poly_basis(self, x):
        
        askey_wilson_basis = [
            tf.ones_like(x)
        ]

        if self.degree > 0:
            askey_wilson_basis.append((2 * (1 + self.a * self.b * self.q) * x - (self.a + self.b) * (1 + self.c * self.d * self.q)) / (1 + self.a * self.b * self.c * self.d * self.q**2))

        for n in range(2, self.degree + 1):
            An = (1 - self.a * self.b * self.q**(n-1)) * (1 - self.c * self.d * self.q**(n-1)) * (1 - self.a * self.b * self.c * self.d * self.q**(2*n-2))
            An /= (1 - self.a * self.b * self.c * self.d * self.q**(2*n-1)) * (1 - self.a * self.b * self.c * self.d * self.q**(2*n))
            Cn = (1 - self.q**n) * (1 - self.a * self.b * self.q**(n-1)) * (1 - self.c * self.d * self.q**(n-1)) * (1 - self.a * self.b * self.c * self.d * self.q**(2*n-2))
            Cn /= (1 - self.a * self.b * self.c * self.d * self.q**(2*n-2)) * (1 - self.a * self.b * self.c * self.d * self.q**(2*n-1))
            askey_wilson_basis.append(
                ((2 * x - An) * askey_wilson_basis[n-1] - Cn * askey_wilson_basis[n-2]) / (1 - self.q**n)
            )

        return tf.stack(askey_wilson_basis, axis=-1) 