import tensorflow as tf
import numpy as np
from arnold.layers.wavelet.wavelet_base import WaveletBase


tfk = tf.keras
tfkl = tfk.layers


class Morelet(WaveletBase):
    """
    Kolmogorov-Arnold Network layer using Morelet wavelets.
    """

    def __init__(self, input_dim, output_dim, omega0_init=5.0):

        super().__init__(input_dim, output_dim)

        self.omega0 = self.add_weight(
            initializer=tfk.initializers.Constant(
                value=omega0_init
            ),
            name='central_frequency',
            trainable=True
        )

    def call(self, inputs):

        x = tf.math.divide(
            tf.math.subtract(
                # (batch_size, 1, self.input_dim),
                tf.expand_dims(inputs, axis=1),
                # (batch_size, output_dim, input_dim)
                tf.tile(
                    tf.expand_dims(self.translation, axis=0), 
                    [tf.shape(inputs)[0], 1, 1]
                )   
            ),
            # (batch_size, output_dim, input_dim)
            tf.tile(
                tf.expand_dims(self.scale, axis=0), 
                [tf.shape(inputs)[0], 1, 1]
            )             
        )

        real = tf.cos(self.omega0 * x)
        envelope = tf.exp(-0.5 * x ** 2)
        wavelet = envelope * real
        
        wavelet_weighted = wavelet * tf.tile(
            tf.expand_dims(self.wavelet_weights, axis=0),
            [tf.shape(wavelet)[0], 1, 1]
        )

        return tf.reduce_sum(wavelet_weighted, axis=-1)
