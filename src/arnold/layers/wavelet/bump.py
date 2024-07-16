import tensorflow as tf
import numpy as np
from arnold.layers.wavelet.wavelet_base import WaveletBase

tfk = tf.keras
tfkl = tfk.layers


class Bump(WaveletBase):
        
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

        

        # Bump wavelet is only defined in the interval (-1, 1)
        x = tf.tanh(x)
#        inside_interval = (x > -1.0) & (x < 1.0)
        
        wavelet = tf.exp(-1.0 / (1 - x**2)) #* tf.cast(inside_interval, x.dtype)

        wavelet_weighted = wavelet * tf.tile(
            tf.expand_dims(self.wavelet_weights, axis=0),
            [tf.shape(wavelet)[0], 1, 1]
        )
        
        return tf.reduce_sum(wavelet_weighted, axis=-1)