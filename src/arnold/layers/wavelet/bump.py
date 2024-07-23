import tensorflow as tf
from arnold.layers.wavelet.wavelet_base import WaveletBase

tfk = tf.keras
tfkl = tfk.layers


class Bump(WaveletBase):

    @tf.function        
    def get_wavelets(self, x):
        return tf.exp(-1.0 / (1 - x**2))
