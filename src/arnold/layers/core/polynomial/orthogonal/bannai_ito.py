import tensorflow as tf
import numpy as np

from ..poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="BannaiIto")
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
        """
        :param input_dim: This layers input size
        :type input_dim: int
        :param output_dim: This layers output size
        :type output_dim: int
        :param degree: The maximum degree of the polynomial basis element (default is 3).
        :type degree: int
        :param decompose_weights: Whether or not to represent the polynomial_coefficients weights tensor as a learnable Tucker decomposition. Default to False.
        :type decompose_weights: bool
        :param core_ranks: A 3-tuple of non-zero, positive integers giving the ranks of the Tucker decomposition core tensor. Ignored if `decompose_weights` is False; defaults to None.
        :type core_ranks: None | Tuple[int, int, int]
        :param tanh_x: Flag indicating whether to normalize any input to [-1, 1] using tanh before further processing.
        :type tanh_x: bool
        :param a_init: Initial value for the a parameter of the BannaiIto polynomials. Defaults to None (a initialized to RandomNormal).
        "type a_init: float | None = None
        :param a_trainable: Flag indicating whether a is a trainable parameter. Defaults to True
        "type a_trainable: bool
        :param b_init: Initial value for the b parameter of the BannaiIto polynomials. Defaults to None (a initialized to RandomNormal).
        "type b_init: float | None = None
        :param b_trainable: Flag indicating whether b is a trainable parameter. Defaults to True
        "type b_trainable: bool
        :param c_init: Initial value for the c parameter of the BannaiIto polynomials. Defaults to None (a initialized to RandomNormal).
        "type c_init: float | None = None
        :param c_trainable: Flag indicating whether c is a trainable parameter. Defaults to True
        "type c_trainable: bool
        """ 
        super().__init__(*args, **kwargs)

        self.a_init = a_init
        self.a_trainable = a_trainable
        self.b_init = b_init
        self.b_trainable = b_trainable
        self.c_init = c_init
        self.c_trainable = c_trainable

        self.a = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.a_init) if self.a_init else tfk.initializers.Zeros(),
            name='a',
            trainable=self.a_trainable
        )

        self.b = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.b_init)  if self.b_init else tfk.initializers.Zeros(),
            name='b',
            trainable=self.b_trainable
        )

        self.c = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.c_init) if self.c_init else tfk.initializers.Zeros(),
            name='c',
            trainable=self.c_trainable
        )

    @tf.function 
    def poly_basis(self, x):
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "a_init": self.a_init,
            "a_trainable": self.a_trainable,
            "b_init": self.b_init,
            "b_trainable": self.b_trainable,
            "c_init": self.c_init,
            "c_trainable": self.c_trainable,
        })
        return config
