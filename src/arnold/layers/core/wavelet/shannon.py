import tensorflow as tf
import numpy as np
from .wavelet_base import WaveletBase

tfk = tf.keras
tfkl = tfk.layers


class Shannon(WaveletBase):
    """
    Kolmogorov-Arnold Network layer using Shannon wavelets.
    """

    @tf.function
    def get_wavelets(self, x):
        # See: https://mathworld.wolfram.com/SincFunction.html
        indices = tf.where(tf.equal(x, 0))
        x = tf.math.divide_no_nan(tf.math.sin(x), x)
        sinc_x = tf.tensor_scatter_nd_update(x, indices, tf.ones((tf.shape(indices)[0], )))
        
        # Applying a Hamming window to limit the infinite support of the sinc function
        window = tf.signal.hamming_window(tf.shape(x)[-1], periodic=False, dtype=x.dtype)
        
        # Shannon wavelet is the product of the sinc function and the window
        return (sinc_x * window)
