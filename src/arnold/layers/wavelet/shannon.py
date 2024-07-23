import tensorflow as tf
import numpy as np
from arnold.layers.wavelet.wavelet_base import WaveletBase

tfk = tf.keras
tfkl = tfk.layers


class Shannon(WaveletBase):

    @tf.function
    def get_wavelets(self, x):
        sinc = tf.experimental.numpy.sinc(x / np.pi) 
        
        # Applying a Hamming window to limit the infinite support of the sinc function
        window = tf.signal.hamming_window(tf.shape(x)[-1], periodic=False, dtype=x.dtype)
        
        # Shannon wavelet is the product of the sinc function and the window
        return (sinc * window)
