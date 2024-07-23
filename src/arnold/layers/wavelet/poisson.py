import tensorflow as tf
import numpy as np
from arnold.layers.wavelet.wavelet_base import WaveletBase

tfk = tf.keras
tfkl = tfk.layers


class Poisson(WaveletBase):
    """
    Kolmogorov-Arnold Network layer using Poisson wavelets. 
    """

    @tf.function
    def get_wavelets(self, x):
        return (1 / np.pi) * tf.math.divide(
            tf.math.subtract(1.0, tf.square(x)),
            tf.square(tf.math.add(1.0, tf.square(x)))
        )
