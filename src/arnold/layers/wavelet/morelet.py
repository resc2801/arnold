import tensorflow as tf
from arnold.layers.wavelet.wavelet_base import WaveletBase


tfk = tf.keras
tfkl = tfk.layers


class Morelet(WaveletBase):
    """
    Kolmogorov-Arnold Network layer using Morelet wavelets.
    """

    def __init__(self, 
                 input_dim, output_dim, 
                 omega0=5.0, omega_trainable=True):

        super().__init__(input_dim, output_dim)

        self.omega0 = self.add_weight(
            initializer=tfk.initializers.Constant(
                value=omega0
            ),
            name='central_frequency',
            trainable=omega_trainable
        )

    @tf.function
    def get_wavelets(self, x):
        real = tf.cos(self.omega0 * x)
        envelope = tf.exp(-0.5 * x ** 2)
        return (envelope * real)
