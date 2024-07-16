import tensorflow as tf
import numpy as np
from arnold.layers.wavelet.wavelet_base import WaveletBase

tfk = tf.keras
tfkl = tfk.layers


class Shannon(WaveletBase):

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

        sinc = tf.experimental.numpy.sinc(x / np.pi) 
        
        # Applying a Hamming window to limit the infinite support of the sinc function
        window = tf.signal.hamming_window(tf.shape(x)[-1], periodic=False, dtype=x.dtype)
        
        # Shannon wavelet is the product of the sinc function and the window
        wavelet = sinc * window

        wavelet_weighted = wavelet * tf.tile(
            tf.expand_dims(self.wavelet_weights, axis=0),
            [tf.shape(wavelet)[0], 1, 1]
        )
        
        return tf.reduce_sum(wavelet_weighted, axis=-1)
