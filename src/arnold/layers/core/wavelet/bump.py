import tensorflow as tf
from .wavelet_base import WaveletBase

tfk = tf.keras
tfkl = tfk.layers


class Bump(WaveletBase):
    """
    NOTE: Defunct
    """

    @tf.function        
    def get_wavelets(self, x):
        inside_interval = (x > -1.0) & (x < 1.0)
        return (tf.exp(-1.0 / (1 - x**2)) *  inside_interval)
