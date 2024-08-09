import tensorflow as tf
import numpy as np

from ..poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="AssociatedMeixnerPollaczek")
class AssociatedMeixnerPollaczek(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Associated Meixner–Pollaczek polynomials.

    Meixner-Pollaczek polynomials are orthogonal on the real line with respect to the weight function given by the Meixner-Pollaczek distribution. 
    The Associated Meixner–Pollaczek polynomials are generated by the three-term recurrence relation:

    * :math:`P^{\lambda}_{-1}(x; \phi, c) = 0`
    * :math:`P^{\lambda}_{0}(x; \phi, c)  = 1`
    * :math:`P^{\lambda}_{n+1}(x; \phi, c) = \frac{(2 * x * \sin(\phi) + 2*(n + c + \lambda)* P^{\lambda}_{n}(x; \phi, c) - (n + c + 2*\lambda - 1) * P^{\lambda}_{n-1}(x; \phi, c)}{n + c + 1}, \, n >= 0`

    See also: https://dlmf.nist.gov/18.30#v
    """

    def __init__(
            self, 
            *args,
            lambda_init: float | None = None, lambda_trainable=True, 
            phi_init: float | None = None, phi_trainable=True, 
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

        :param lambda_init: Initial value for the lambda parameter of the AssociatedMeixnerPollaczek polynomials. Defaults to None (a initialized to RandomNormal).
        :type lambda_init: float | None = None

        :param lambda_trainable: Flag indicating whether lambda is a trainable parameter. Defaults to True
        :type lambda_trainable: bool

        :param phi_init: Initial value for the phi parameter of the AssociatedMeixnerPollaczek polynomials. Defaults to None (a initialized to RandomNormal).
        :type phi_init: float | None = None

        :param phi_trainable: Flag indicating whether phi is a trainable parameter. Defaults to True
        :type phi_trainable: bool

        :param c_init: Initial value for the c parameter of the AssociatedMeixnerPollaczek polynomials. Defaults to None (a initialized to RandomNormal).
        :type c_init: float | None = None

        :param c_trainable: Flag indicating whether c is a trainable parameter. Defaults to True
        :type c_trainable: bool
        """    
        super().__init__(*args, **kwargs)

        self.lambda_init = lambda_init
        self.lambda_trainable = lambda_trainable
        self.phi_init = phi_init
        self.phi_trainable = phi_trainable
        self.c_init = c_init
        self.c_trainable = c_trainable

        self.lambda_ = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.lambda_init) if self.lambda_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0),
            name='lambda',
            trainable=self.lambda_trainable
        )

        self.phi = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.phi_init) if self.phi_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0),
            name='phi',
            trainable=self.phi_trainable
        )

        self.c = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.c_init) if self.c_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0),
            name='c',
            trainable=self.c_trainable
        )

    @tf.function
    def pseudo_vandermonde(self, x):
        # :math:`P^{\lambda}_{0}(x; \phi, c)  = 1`
        meixner_pollaczek_basis = [ tf.ones_like(x) ]

        if self.degree > 0:
            # \frac{(2 * x * \sin(\phi) + 2*(n + c + \lambda)}{n + c + 1} 
            meixner_pollaczek_basis.append(
                (2 * x * tf.sin(self.phi) + 2 * (1 + self.c + self.lambda_) * tf.cos(self.phi)) / (1 + self.c + 1.0)
            )

        for n in range(2, self.degree + 1):
            # :math:`P^{\lambda}_{n+1}(x; \phi, c) = \frac{(2 * x * \sin(\phi) + 2*(n + c + \lambda)* P^{\lambda}_{n}(x; \phi, c) - (n + c + 2*\lambda - 1) * P^{\lambda}_{n-1}(x; \phi, c)}{n + c + 1}  when n >= 2
            term1 = (2 * x * tf.sin(self.phi) + 2 * (n + self.c + self.lambda_) * tf.cos(self.phi))
            term2 = (n + self.c + 2 * self.lambda_ - 1.0)
            term3 = (n + self.c + 1.0)
            meixner_pollaczek_basis.append(
                (term1 * meixner_pollaczek_basis[n-1] - term2 * meixner_pollaczek_basis[n-2]) / term3
            )
        
        return tf.stack(meixner_pollaczek_basis, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "lambda_init": self.lambda_init,
            "lambda_trainable": self.lambda_trainable,
            "phi_init": self.phi_init,
            "phi_trainable": self.phi_trainable,
            "c_init": self.c_init,
            "c_trainable": self.c_trainable,
        })
        return config
