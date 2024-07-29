import tensorflow as tf
import numpy as np

from ..poly_base import PolynomialBase
from arnold.math import generalized_hypergeometric

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="Wilson")
class Wilson(PolynomialBase):
    """
    Kolmogorov-Arnold Network layer using the Wilson polynomials.
    """

    def __init__(
            self, 
            *args,
            a_init: float | None = None, a_trainable=True, 
            b_init: float | None = None, b_trainable=True,
            c_init: float | None = None, c_trainable=True, 
            d_init: float | None = None, d_trainable=True, 
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

        :param a_init: Initial value for the a parameter of the Wilson polynomials. Must be a positive, non-zero float; defaults to None (a initialized to RandomNormal).
        :type a_init: float | None = None

        :param a_trainable: Flag indicating whether a is a trainable parameter. Defaults to True
        :type a_trainable: bool

        :param b_init: Initial value for the b parameter of the Wilson polynomials. Must be a positive, non-zero float; defaults to None (a initialized to RandomNormal).
        :type b_init: float | None = None

        :param b_trainable: Flag indicating whether b is a trainable parameter. Defaults to True
        :type b_trainable: bool

        :param c_init: Initial value for the c parameter of the Wilson polynomials. Must be a positive, non-zero float; defaults to None (a initialized to RandomNormal).
        :type c_init: float | None = None

        :param c_trainable: Flag indicating whether c is a trainable parameter. Defaults to True
        :type c_trainable: bool

        :param d_init: Initial value for the parameter d of the Wilson polynomials. Must be a positive, non-zero float; defaults to None (a initialized to RandomNormal).
        :type d_init: float | None = None

        :param d_trainable: Flag indicating whether d is a trainable parameter. Defaults to True
        :type d_trainable: bool
        """ 
        super().__init__(*args, **kwargs)

        self.a_init = a_init
        self.a_trainable = a_trainable
        self.b_init = b_init
        self.b_trainable = b_trainable
        self.c_init = c_init
        self.c_trainable = c_trainable
        self.d_init = d_init
        self.d_trainable = d_trainable

        self.a = self.add_weight(
            initializer=tfk.initializers.Constant(value=tf.math.log(self.a_init)) if self.a_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='a_logits',
            trainable=self.a_trainable
        )

        self.b = self.add_weight(
            initializer=tfk.initializers.Constant(value=tf.math.log(self.b_init)) if self.b_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='b_logits',
            trainable=self.b_trainable
        )

        self.c = self.add_weight(
            initializer=tfk.initializers.Constant(value=tf.math.log(self.c_init)) if self.c_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='c_logits',
            trainable=self.c_trainable
        )

        self.d = self.add_weight(
            initializer=tfk.initializers.Constant(value=tf.math.log(self.d_init)) if self.d_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='d_logits',
            trainable=self.d_trainable
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "a_init": self.a_init,
            "a_trainable": self.a_trainable,
            "b_init": self.b_init,
            "b_trainable": self.b_trainable,
            "c_init": self.c_init,
            "c_trainable": self.c_trainable,
            "d_init": self.d_init,
            "d_trainable": self.d_trainable,
        })
        return config


    @tf.function
    def poly_basis(self, x):

        # recover parameters from logits
        a = tf.math.exp(self.a)
        b = tf.math.exp(self.b)
        c = tf.math.exp(self.c)
        d = tf.math.exp(self.d)
        n = self.degree + 1

        return tf.stack(
            list(
                map(
                    lambda degree: generalized_hypergeometric(
                        [-n * tf.ones_like(x), (a+b+c+d+n-1)* tf.ones_like(x), a* tf.ones_like(x) - x, a* tf.ones_like(x) + x], 
                        [(a+b) * tf.ones_like(x), (a+c) * tf.ones_like(x), (a+d) * tf.ones_like(x)], 
                        1.0, 
                        n + 1),
                    range(n)
                )
            ),
            axis=-1
        )
