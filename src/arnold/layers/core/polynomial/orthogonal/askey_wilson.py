import tensorflow as tf
import numpy as np

from ..poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="AskeyWilson")
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

        :param a_init: Initial value for the a parameter of the AskeyWilson polynomials. Defaults to None (a initialized to RandomNormal).
        :type a_init: float | None = None

        :param a_trainable: Flag indicating whether a is a trainable parameter. Defaults to True
        :type a_trainable: bool

        :param b_init: Initial value for the b parameter of the AskeyWilson polynomials. Defaults to None (a initialized to RandomNormal).
        :type b_init: float | None = None

        :param b_trainable: Flag indicating whether b is a trainable parameter. Defaults to True
        :type b_trainable: bool

        :param c_init: Initial value for the c parameter of the AskeyWilson polynomials. Defaults to None (a initialized to RandomNormal).
        :type c_init: float | None = None

        :param c_trainable: Flag indicating whether c is a trainable parameter. Defaults to True
        :type c_trainable: bool

        :param d_init: Initial value for the parameter d of the AskeyWilson polynomials. Defaults to None (a initialized to RandomNormal).
        :type d_init: float | None = None

        :param d_trainable: Flag indicating whether d is a trainable parameter. Defaults to True
        :type d_trainable: bool

        :param q_init: Initial value for the q parameter of the AskeyWilson polynomials. Defaults to None (q initialized to RandomNormal).
        :type q_init: float | None = None
        
        :param q_trainable: Flag indicating whether q is a trainable parameter. Defaults to True
        :type q_trainable: bool
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
        self.q_init = q_init
        self.q_trainable = q_trainable

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

        self.c = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.c_init) if self.c_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='c',
            trainable=self.c_trainable
        )

        self.d = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.d_init) if self.d_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='d',
            trainable=self.d_trainable
        )

        self.q = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.q_init) if self.q_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='q',
            trainable=self.q_trainable
        )

    @tf.function(jit_compile=True)
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
            "q_init": self.q_init,
            "q_trainable": self.q_trainable,
        })
        return config
