from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

tfk = tf.keras
tfkl = tfk.layers

@tfk.utils.register_keras_serializable(package="arnold", name="PolynomialBase")
class PolynomialBase(tfkl.Layer, ABC):
    """
    Abstract base class for Kolmogorov-Arnold Network layer using polynomial basis.
    """

    def __init__(
            self, 
            input_dim, output_dim, 
            degree,
            *args,
            tanh_x=True,
            **kwargs):
        
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim 
        self.output_dim = output_dim
        self.degree = degree
        self.tanh_x = tanh_x

        self.poly_coeffs = self.add_weight(
            shape=(self.input_dim, self.output_dim, self.degree + 1),
            initializer=tfk.initializers.RandomNormal(
                mean=0.0, 
                stddev=(1.0 / (self.input_dim * (self.degree + 1)))
            ),
            constraint=None,
            regularizer=None,
            trainable=True,
            name='polynomial_coefficients'     
        )

    def call(self, inputs):
        # Normalize x to [-1, 1] using tanh
        x = tf.tanh(inputs) if self.tanh_x else inputs
        x = tf.reshape(x, (-1, self.input_dim))

        # Compute the polynom interpolation with y.shape=(batch_size, output_dim)
        y = tf.einsum(
            'bid,iod->bo', 
            self.poly_basis(x), 
            self.poly_coeffs
        )
        
        return y

    @abstractmethod
    def poly_basis(self, x):
        pass

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "degree": self.degree,
            "tanh_x": self.tanh_x
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
 