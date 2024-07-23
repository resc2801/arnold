import tensorflow as tf
import numpy as np
from arnold.layers.wavelet.wavelet_base import WaveletBase

tfk = tf.keras
tfkl = tfk.layers


class Meyer(WaveletBase):
    """
    Kolmogorov-Arnold Network layer using Meyer wavelets. 
    """

    @tf.function
    def get_wavelets(self, x):
        v = tf.abs(x)
        return tf.sin(np.pi * v) * self.__meyer_aux(v)

    @tf.function
    def __nu(self, t):
        return t**4 * (35 - 84*t + 70*t**2 - 20*t**3)

    @tf.function        
    def __meyer_aux(self, v):
        return tf.where(
            v <= 1/2,
            tf.ones_like(v),
            tf.where(
                v >= 1,
                tf.zeros_like(v),
                tf.cos(np.pi / 2 * self.__nu(2 * v - 1))
            )
        )
