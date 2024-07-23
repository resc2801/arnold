import tensorflow as tf
import numpy as np
from arnold.layers.radial.rbf_base import RBFBase

tfk = tf.keras
tfkl = tfk.layers

class InverseQuadraticRBF(RBFBase):
    """
    Kolmogorov-Arnold Network layer using a inverse quadratic radial basis function.
    """

    def get_basis(self, x):
        basis = 1 / (1 + ((x - self.grid) / ((self.grid_max - self.grid_min) / (self.num_grids - 1))) ** 2)
        basis = tf.reshape(basis, (-1, self.input_dim * self.num_grids))
        return basis
