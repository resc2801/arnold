import tensorflow as tf
import numpy as np

from arnold.layers.polynomial.poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


class BannaiIto(PolynomialBase):
    """
    Kolmogorov-Arnold Network layer using the Bannai-Ito polynomials.
    """

    def __init__(
            self, 
            *args,
            a_init: float | None = None, a_trainable=True, 
            b_init: float | None = None, b_trainable=True, 
            c_init: float | None = None, c_trainable=True,
            **kwargs):
        
        super().__init__(*args, **kwargs)

        self.a = self.add_weight(
            initializer=tfk.initializers.Constant(value=a_init) if a_init else tfk.initializers.Zeros(),
            name='a',
            trainable=a_trainable
        )

        self.b = self.add_weight(
            initializer=tfk.initializers.Constant(value=b_init)  if b_init else tfk.initializers.Zeros(),
            name='b',
            trainable=b_trainable
        )

        self.c = self.add_weight(
            initializer=tfk.initializers.Constant(value=c_init) if c_init else tfk.initializers.Zeros(),
            name='c',
            trainable=c_trainable
        )

    @tf.function 
    def poly_basis(self, x):
        """
        Evaluate Bannai-Ito basis polynomials for given `x`.
        """

        bannai_ito_basis = [tf.ones_like(x)]

        if self.degree > 0:
            bannai_ito_basis.append(
                (x - self.a) / (self.b + self.c + 1.0)
            )

        for n in range(2, self.degree + 1):
            An = (2 * n + self.b + self.c - 1) * (2 * n + self.b + self.c) / (2 * (n + self.b + self.c))
            Cn = -(n + self.b - 1) * (n + self.c - 1) / (2 * (n + self.b + self.c))

            bannai_ito_basis.append(
                ((x - An) * bannai_ito_basis[n-1] - Cn * bannai_ito_basis[n - 2]) / (n + self.b + self.c)
            )

        return tf.stack(bannai_ito_basis, axis=-1)
