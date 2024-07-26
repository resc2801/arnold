import tensorflow as tf
import numpy as np
from .rbf_base import RBFBase

tfk = tf.keras
tfkl = tfk.layers

class GaussianRBF(RBFBase):
    """
    Kolmogorov-Arnold Network layer using a gaussian radial basis function.
    """

    @tf.function
    def get_basis(self, x):
        basis = tf.exp(-((x - self.grid) / ((self.grid_max - self.grid_min) / (self.num_grids - 1))) ** 2)
        basis = tf.reshape(basis, (-1, self.input_dim * self.num_grids))
        return basis
