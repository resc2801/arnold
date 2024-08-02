import tensorflow as tf
from .wavelet_base import WaveletBase

tfk = tf.keras
tfkl = tfk.layers


class DerivativeOfGaussian(WaveletBase):
    """
    Kolmogorov-Arnold Network layer using DerivativeOfGaussian wavelets.
    """

    @tf.function        
    def get_wavelets(self, x):
        return -x * tf.exp(-0.5 * x**2)
