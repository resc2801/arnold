from abc import abstractmethod
from typing import Tuple
import tensorflow as tf

from arnold.layers.core.kan_base import KANBase

tfk = tf.keras
tfkl = tfk.layers

@tfk.utils.register_keras_serializable(package="arnold", name="PolynomialBase")
class PolynomialBase(KANBase):
    """
    Abstract base class for Kolmogorov-Arnold Network layer using polynomial basis.
    """

    def __init__(
            self, 
            degree:int,
            *args,
            core_ranks:None | Tuple[int, int, int] = None,
            **kwargs):
        """
        :param input_dim: This layers input size
        :type input_dim: int

        :param output_dim: This layers output size
        :type output_dim: int

        :param degree: The maximum degree of the polynomial basis element (default is 3).
        :type degree: int

        :param core_ranks: A 3-tuple of non-zero, positive integers giving the ranks of the Tucker decomposition for the polynomial_coefficients weights tensor. Ignored if `decompose_weights` is False; defaults to None.
        :type core_ranks: None | Tuple[int, int, int]

        :param tanh_x: Flag indicating whether to normalize any input to [-1, 1] using tanh before further processing.
        :type tanh_x: bool
        """
        super().__init__(*args, **kwargs)

        self.degree = degree
        self.core_ranks = core_ranks

        if self.core_ranks is not None:
            self.r1, self.r2, self.r3 = core_ranks

            self.poly_coeffs_core = self.add_weight(
                shape=(self.r1, self.r2, self.r3),
                initializer=tfk.initializers.RandomNormal(
                    mean=0.0, 
                    stddev=(1.0 / (self.input_dim * (self.degree + 1)))
                ),
                constraint=None,
                regularizer=None,
                trainable=True,
                name='polynomial_coefficients_core'     
            )

            self.poly_coeffs_A = self.add_weight(
                shape=(self.input_dim, self.r1),
                initializer=tfk.initializers.RandomNormal(
                    mean=0.0, 
                    stddev=(1.0 / (self.input_dim * (self.degree + 1)))
                ),
                constraint=None,
                regularizer=None,
                trainable=True,
                name='polynomial_coefficients_input_dim'     
            )

            self.poly_coeffs_B = self.add_weight(
                shape=(self.degree + 1, self.r2),
                initializer=tfk.initializers.RandomNormal(
                    mean=0.0, 
                    stddev=(1.0 / (self.input_dim * (self.degree + 1)))
                ),
                constraint=None,
                regularizer=None,
                trainable=True,
                name='polynomial_coefficients_degree'     
            )

            self.poly_coeffs_C = self.add_weight(
                shape=(self.output_dim, self.r3),
                initializer=tfk.initializers.RandomNormal(
                    mean=0.0, 
                    stddev=(1.0 / (self.input_dim * (self.degree + 1)))
                ),
                constraint=None,
                regularizer=None,
                trainable=True,
                name='polynomial_coefficients_output_dim'     
            )

        else:
            self.poly_coeffs = self.add_weight(
                shape=(self.input_dim, self.degree + 1, self.output_dim),
                initializer=tfk.initializers.RandomNormal(
                    mean=0.0, 
                    stddev=(1.0 / (self.input_dim * (self.degree + 1)))
                ),
                constraint=None,
                regularizer=None,
                trainable=True,
                name='polynomial_coefficients'     
            )

    @tf.function(jit_compile=True)
    def call(self, inputs):
        """
        Performs the forward computation

        :param inputs: Data to perform the forward computation on.
        :type inputs: tf.Tensor
        """
        # Always flatten any given input first!
        x = tf.reshape(inputs, (-1, self.input_dim))

        # Normalize x to [-1, 1] using tanh if requested 
        x = tf.tanh(x) if self.tanh_x else x

        # Compute the polynom interpolation with y.shape=(batch_size, output_dim)
        if self.core_ranks is not None:
            y = tf.einsum(
            'bid,xyz,ix,oy,dz->bo', 
                self.pseudo_vandermonde(x), self.poly_coeffs_core, self.poly_coeffs_A, self.poly_coeffs_B, self.poly_coeffs_C,
                optimize='auto'
            )
        else:
            y = tf.einsum(
                'bid,ido->bo', 
                self.pseudo_vandermonde(x), 
                self.poly_coeffs,
                optimize='auto'
            )
        
        return y

    @abstractmethod
    def pseudo_vandermonde(self, x):
        """
        Computes the pseudo-Vandermonde matrix for given `x`.

        :param x: Data to compute the pseudo-vandermonde tensor with.
        :type x: tf.Tensor

        :returns: pseudo-vandermonde tensor
        :rtype: tf.Tensor
        """
        raise NotImplementedError(
            f"Layer {self.__class__.__name__} does not have a "
            "`pseudo_vandermonde()` method implemented."
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "degree": self.degree,
        })
        return config
