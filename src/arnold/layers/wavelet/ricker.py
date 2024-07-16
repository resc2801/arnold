import tensorflow as tf
import numpy as np
from arnold.layers.wavelet.wavelet_base import WaveletBase

tfk = tf.keras
tfkl = tfk.layers


class Ricker(WaveletBase):
    """
    Kolmogorov-Arnold Network layer using Ricker (Mexican hat) wavelets.
    """

    def __init__(self, input_dim, output_dim, sigma=1.0, sigma_trainable=True):

        super().__init__(input_dim, output_dim)

        self.sigma = self.add_weight(
            initializer=tfk.initializers.Constant(
                value=sigma
            ),
            name='standard_deviation',
            trainable=sigma_trainable
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

        term1 = (1 - ((x / self.sigma) ** 2))
        term2 = tf.exp(-0.5 * (x / self.sigma) ** 2)
        wavelet = (2 / (tf.math.sqrt(3.0) * np.pi**0.25)) * term1 * term2
        
        wavelet_weighted = wavelet * tf.tile(
            tf.expand_dims(self.wavelet_weights, axis=0),
            [tf.shape(wavelet)[0], 1, 1]
        )
        
        return tf.reduce_sum(wavelet_weighted, axis=-1)
