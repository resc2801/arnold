from abc import abstractmethod
import tensorflow as tf

from arnold.layers.core.kan_base import KANBase

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="WaveletBase")
class WaveletBase(KANBase):
    """
    Abstract base class for Kolmogorov-Arnold Network layer using wavelets.
    """

    def __init__(self, 
                 *args,
                 **kwargs):
        """
        :param input_dim: This layers input size
        :type input_dim: int

        :param output_dim: This layers output size
        :type output_dim: int

        :param tanh_x: Flag indicating whether to normalize any input to [-1, 1] using tanh before further processing.
        :type tanh_x: bool
        """
        super().__init__(*args, **kwargs)
        
        # Parameters for wavelet transformation
        self.scale = self.add_weight(
            shape=(self.output_dim, self.input_dim),
            initializer=tfk.initializers.Ones(),
            name='scale',
            trainable=True
        )

        self.translation = self.add_weight(
            shape=(self.output_dim, self.input_dim),
            initializer=tfk.initializers.Zeros(),
            name='translation',
            trainable=True
        )

        # Linear weights for combining outputs
        self.wavelet_weights = self.add_weight(
            shape=(self.output_dim, self.input_dim),
            initializer=tfk.initializers.HeUniform(),
            name='wavelet_weights',
            trainable=True
        )

    @tf.function
    def call(self, inputs):
        # Normalize x to [-1, 1] using tanh
        x = tf.tanh(inputs) if self.tanh_x else inputs

        x = tf.math.divide(
            tf.math.subtract(
                # (batch_size, 1, self.input_dim),
                tf.expand_dims(x, axis=1),
                # (1, output_dim, input_dim)
                tf.expand_dims(self.translation, axis=0)
            ),
            # (1, output_dim, input_dim)
            tf.expand_dims(self.scale, axis=0), 
        )

        # (batch_size, output_dim, input_dim)
        wavelets = self.get_wavelets(x)

        wavelets_weighted = wavelets * self.wavelet_weights
        
        return tf.reduce_sum(wavelets_weighted, axis=-1)

    @abstractmethod
    def get_wavelets(self, x):
        """
        Computes the wavelets for given `x`.

        :param x: Data to compute the wavelets for.
        :type x: tf.Tensor

        :returns: wavelets
        :rtype: tf.Tensor
        """
        raise NotImplementedError(
            f"Layer {self.__class__.__name__} does not have a "
            "`get_wavelets()` method implemented."
        )
