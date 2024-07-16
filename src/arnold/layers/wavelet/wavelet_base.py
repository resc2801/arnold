from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

tfk = tf.keras
tfkl = tfk.layers


class WaveletBase(tfkl.Layer, ABC):
    """
    Abstract base class for Kolmogorov-Arnold Network layer using wavelets.
    """

    def __init__(self, input_dim, output_dim):

        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Parameters for wavelet transformation
        self.scale = self.add_weight(
            shape=(self.output_dim, self.input_dim),
            initializer=tfk.initializers.Ones(),
            name='scale',
            trainable=True
        )

        self.translation = self.add_weight(
            shape=(self.output_dim, self.input_dim),
            initializer=tfk.initializers.Zeros(),
            name='translation',
            trainable=True
        )

        # Linear weights for combining outputs
        self.wavelet_weights = self.add_weight(
            shape=(self.output_dim, self.input_dim),
            initializer=tfk.initializers.HeUniform(),
            name='wavelet_weights',
            trainable=True
        )

    @abstractmethod
    def call(self, inputs):
        pass

