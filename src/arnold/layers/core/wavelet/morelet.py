import tensorflow as tf
from .wavelet_base import WaveletBase


tfk = tf.keras
tfkl = tfk.layers


class Morelet(WaveletBase):
    """
    Kolmogorov-Arnold Network layer using Morelet wavelets.
    """

    def __init__(self, 
                 *args,
                 omega_init:float = 5.0, 
                 omega_trainable=True,
                 **kwargs):
        """
        :param input_dim: This layers input size
        :type input_dim: int

        :param output_dim: This layers output size
        :type output_dim: int

        :param scale_init: Initial non-zero, positive value for the wavelet scale parameter; defaults to None (log(scale) initialized to HeNormal).
        :type scale_init: non-zero, positive float | None = None

        :param scale_trainable: Flag indicating whether scale is a trainable parameter. Defaults to True
        :type scale_trainable: bool

        :param translation_init: Initial translation value for the wavelet scale parameter; defaults to None (initialized to HeNormal).
        :type translation_init: float | None = None

        :param translation_trainable: Flag indicating whether translation is a trainable parameter. Defaults to True
        :type translation_trainable: bool

        :param omega_init: Initial value for the omega parameter of the Morelet wavelet. Defaults to 5.0.
        :type omega_init: float

        :param omega_trainable: Flag indicating whether omega is a trainable parameter. Defaults to True
        "type omega_trainable: bool

        :param tanh_x: Flag indicating whether to normalize any input to [-1, 1] using tanh before further processing.
        :type tanh_x: bool
        """
        super().__init__(*args, **kwargs)

        self.omega_init = omega_init
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
