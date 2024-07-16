import tensorflow as tf

from arnold.layers.polynomial.poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


class Laurent(PolynomialBase):
    """
    Kolmogorov-Arnold Network layer using Laurent polynomials.

    Laurent polynomials differ from ordinary polynomials in that they may have terms of negative degree. 
    """

    def __init__(
            self, 
            *args,
            **kwargs):
        
        super().__init__(*args, **kwargs)

        self.poly_coeffs = self.add_weight(
            shape=(self.input_dim, self.output_dim, 2 * self.degree + 1),
            initializer=tfk.initializers.RandomNormal(
                mean=0.0, 
                stddev=(1.0 / (self.input_dim * (2 * self.degree + 1)))
            ),
            trainable=True,
            regularizer=None,
            constraint=None,
            name='polynomial_coefficients' 
        )

    @tf.function
    def poly_basis(self, x):

        laurent_basis = tf.stack(
            [tf.pow(x, k) for k in range(-1 * self.degree, self.degree + 1, 1)], 
            axis=-1
        )

        return laurent_basis
