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
        eps = 1e-07
        x = tf.clip_by_value(x, -1.0+eps, 1.0-eps)
        return tf.exp(-1.0 / (1 - x**2)) #*  inside_interval)
