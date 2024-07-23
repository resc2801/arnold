from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

tfk = tf.keras
tfkl = tfk.layers


class RBFBase(tfkl.Layer, ABC):
    """
    Abstract base class for Kolmogorov-Arnold Network layer using radial basis functions.
    """

    def __init__(
            self, 
            input_dim, output_dim, 
            *args,
            grid_min=-2.0, grid_max=2.0, num_grids=8, 
            spline_weight_init_scale=0.1,
            tanh_x=True,
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.spline_weight_init_scale = spline_weight_init_scale
        self.tanh_x = tanh_x

        self.grid = tf.constant(
            tf.linspace(self.grid_min, self.grid_max, self.num_grids),
        )

        self.spline_weight = self.add_weight(
            shape=(self.input_dim * self.num_grids, self.output_dim),
            initializer=tfk.initializers.RandomNormal(
                mean=0.0, 
                stddev=1.0 * self.spline_weight_init_scale
            ),
            trainable=True
        )
    
    def call(self, inputs):
        # Normalize x to [-1, 1] using tanh
        x = tf.tanh(inputs) if self.tanh_x else inputs
        x = tf.reshape(x, (-1, self.input_dim, 1))

        y = tf.matmul(
            self.get_basis(x),
            self.spline_weight
        )

        return y
    
    @abstractmethod
    def get_basis(self, x):
        pass
