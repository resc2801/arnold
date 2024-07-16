import tensorflow as tf
import numpy as np
from arnold.layers.wavelet.wavelet_base import WaveletBase

tfk = tf.keras
tfkl = tfk.layers


class Meyer(WaveletBase):
    """
    Kolmogorov-Arnold Network layer using Meyer wavelets. 
    """

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

        v = tf.abs(x)

        def nu(t):
            return t**4 * (35 - 84*t + 70*t**2 - 20*t**3)
        
        def meyer_aux(v):
            return tf.where(
                v <= 1/2,
                tf.ones_like(v),
                tf.where(
                    v >= 1,
                    tf.zeros_like(v),
                    tf.cos(np.pi / 2 * nu(2 * v - 1))
                )
            )

        wavelet = tf.sin(np.pi * v) * meyer_aux(v)

        wavelet_weighted = wavelet * tf.tile(
            tf.expand_dims(self.wavelet_weights, axis=0),
            [tf.shape(wavelet)[0], 1, 1]
        )
        
        return tf.reduce_sum(wavelet_weighted, axis=-1)
