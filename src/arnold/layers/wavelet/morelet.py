import tensorflow as tf
from arnold.layers.wavelet.wavelet_base import WaveletBase


tfk = tf.keras
tfkl = tfk.layers


class Morelet(WaveletBase):
    """
    Kolmogorov-Arnold Network layer using Morelet wavelets.
    """

    def __init__(self, 
                 *args,
                 omega:float = 5.0, omega_trainable=True,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.omega_init = omega
        self.omega_trainable = omega_trainable

        self.omega0 = self.add_weight(
            initializer=tfk.initializers.Constant(
                value=self.omega_init
            ),
            name='central_frequency',
            trainable=self.omega_trainable
        )

    @tf.function
    def get_wavelets(self, x):
        real = tf.cos(self.omega0 * x)
        envelope = tf.exp(-0.5 * x ** 2)
        return (envelope * real)

    def get_config(self):
        config = super().get_config()
        config.update({
            "omega": self.omega_init,
            "omega_trainable": self.omega_trainable
        })
        return config
