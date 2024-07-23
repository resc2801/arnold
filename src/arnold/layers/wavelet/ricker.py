import tensorflow as tf
import numpy as np
from arnold.layers.wavelet.wavelet_base import WaveletBase

tfk = tf.keras
tfkl = tfk.layers


class Ricker(WaveletBase):
    """
    Kolmogorov-Arnold Network layer using Ricker (Mexican hat) wavelets.
    """

    def __init__(self, 
                 *args,
                 sigma:float=1.0, sigma_trainable=True,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.sigma_init = sigma
        self.sigma_trainable = sigma_trainable

        self.sigma = self.add_weight(
            initializer=tfk.initializers.Constant(
                value=self.sigma_init
            ),
            name='standard_deviation',
            trainable=self.sigma_trainable
        )
    
    @tf.function
    def get_wavelets(self, x):
        term1 = (1 - ((x / self.sigma) ** 2))
        term2 = tf.exp(-0.5 * (x / self.sigma) ** 2)
        return (2 / (tf.math.sqrt(3.0) * np.pi**0.25)) * term1 * term2

    def get_config(self):
        config = super().get_config()
        config.update({
            "sigma": self.sigma_init,
            "sigma_trainable": self.sigma_trainable
        })
        return config
