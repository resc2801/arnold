import tensorflow as tf
import numpy as np


tfk = tf.keras
tfkl = tfk.layers

class InverseMultiQuadricRBF(tfkl.Layer):
    """
    Kolmogorov-Arnold Network layer using a inverse multiquadric radial basis function.
    """

    def __init__(
            self, 
            input_dim, 
            output_dim, 
            grid_min=-2.0, 
            grid_max=2.0, 
            num_grids=8, 
            spline_weight_init_scale=0.1
        ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.spline_weight_init_scale = spline_weight_init_scale

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
        x = tf.reshape(inputs, (-1, self.input_dim, 1))
        basis = 1 / tf.sqrt(1 + ((x - self.grid) / ((self.grid_max - self.grid_min) / (self.num_grids - 1))) ** 2)
        basis = tf.reshape(basis, (-1, self.input_dim * self.num_grids))

        y = tf.matmul(
            basis,
            self.spline_weight
        )

        return y

