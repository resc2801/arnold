import tensorflow as tf
from arnold.layers.wavelet.wavelet_base import WaveletBase

tfk = tf.keras
tfkl = tfk.layers


class Bump(WaveletBase):

    @tf.function        
    def get_wavelets(self, x):
        # Bump wavelet is only defined in the interval (-1, 1)
        # We apply a condition to restrict the computation to this interval
        inside_interval = (x > -1.0) & (x < 1.0)
        return tf.exp(-1.0 / (1 - x**2)) * inside_interval.float()
