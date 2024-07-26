import tensorflow as tf

from .poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="Laurent")
class Laurent(PolynomialBase):
    """
    Kolmogorov-Arnold Network layer using Laurent polynomials.

    Laurent polynomials differ from ordinary polynomials in that they may have terms of negative degree. 
    """

    def __init__(
            self, 
            degree:int,
            *args,
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
        """

        super().__init__(2*degree, *args, **kwargs)
        self.degree = degree

        # self.poly_coeffs = self.add_weight(
        #     shape=(self.input_dim, self.output_dim, 2 * self.degree + 1),
        #     initializer=tfk.initializers.RandomNormal(
        #         mean=0.0, 
        #         stddev=(1.0 / (self.input_dim * (2 * self.degree + 1)))
        #     ),
        #     trainable=True,
        #     regularizer=None,
        #     constraint=None,
        #     name='polynomial_coefficients' 
        # )

    @tf.function
    def poly_basis(self, x):
        laurent_basis = tf.stack(
            [tf.pow(x, k) for k in range(-1 * self.degree, self.degree + 1, 1)], 
            axis=-1
        )

        return laurent_basis
