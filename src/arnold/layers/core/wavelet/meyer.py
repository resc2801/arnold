import tensorflow as tf
import numpy as np
from .wavelet_base import WaveletBase

tfk = tf.keras
tfkl = tfk.layers


class Meyer(WaveletBase):
    """
    Kolmogorov-Arnold Network layer using Meyer wavelets. 
    """

    @tf.function
    def get_wavelets(self, x):

        def __nu(t):
            return t**4 * (35 - 84*t + 70*t**2 - 20*t**3)
        
        def __meyer_aux(v):
            return tf.where(
                v <= 1/2,
                tf.ones_like(v),
                tf.where(
                    v >= 1,
                    tf.zeros_like(v),
                    tf.cos(np.pi / 2 * __nu(2 * v - 1))
                )
            )
        
        v = tf.abs(x)
        
        return tf.sin(np.pi * v) * __meyer_aux(v)
