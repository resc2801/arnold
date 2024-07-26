import tensorflow as tf
import numpy as np
from .wavelet_base import WaveletBase

tfk = tf.keras
tfkl = tfk.layers


class Ricker(WaveletBase):
    """
    Kolmogorov-Arnold Network layer using Ricker (Mexican hat) wavelets.
    """

    def __init__(self, 
                 *args,
                 sigma_init:float=1.0, 
                 sigma_trainable=True,
                 **kwargs):
        """
        :param input_dim: This layers input size
        :type input_dim: int

        :param output_dim: This layers output size
        :type output_dim: int

        :param sigma_init: Initial value for the sigma parameter of the Ricker wavelet. Defaults to 1.0.
        :type sigma_init: float

        :param sigma_trainable: Flag indicating whether sigma is a trainable parameter. Defaults to True
        "type sigma_trainable: bool

        :param tanh_x: Flag indicating whether to normalize any input to [-1, 1] using tanh before further processing.
        :type tanh_x: bool
        """
        super().__init__(*args, **kwargs)

        self.sigma_init = sigma_init
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
