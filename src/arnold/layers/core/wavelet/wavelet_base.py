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
                 scale_init: float | None = None, scale_trainable=True, 
                 translation_init: float | None = None, translation_trainable=True, 
                 **kwargs):
        """
        :param input_dim: This layers input size
        :type input_dim: int

        :param output_dim: This layers output size
        :type output_dim: int

        :param tanh_x: Flag indicating whether to normalize any input to [-1, 1] using tanh before further processing.
        :type tanh_x: bool

        :param scale_init: Initial non-zero, positive value for the wavelet scale parameter; defaults to None (log(scale) initialized to HeNormal).
        :type scale_init: non-zero, positive float | None = None

        :param scale_trainable: Flag indicating whether scale is a trainable parameter. Defaults to True
        :type scale_trainable: bool

        :param translation_init: Initial translation value for the wavelet scale parameter; defaults to None (initialized to HeNormal).
        :type translation_init: float | None = None

        :param translation_trainable: Flag indicating whether translation is a trainable parameter. Defaults to True
        :type translation_trainable: bool
        """
        super().__init__(*args, **kwargs)

        if (scale_init is not None) and (scale_init <= 0):
            raise ValueError('Non-zero, positive value for the initial wavelet scale parameter required!')
        
        self.scale_init = scale_init
        self.translation_init = translation_init
        
        # Parameters for wavelet transformation
        self.scale = self.add_weight(
            shape=(1, self.output_dim, self.input_dim),
            initializer=tfk.initializers.Constant(value=tf.math.log(self.scale_init)) if self.scale_init else tfk.initializers.HeNormal(),
            name='scale_logits',
            trainable=True
        )

        self.translation = self.add_weight(
            shape=(1, self.output_dim, self.input_dim),
            initializer=tfk.initializers.Constant(value=self.translation_init) if self.translation_init else tfk.initializers.HeNormal(),
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

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def call(self, inputs):
        # Normalize x to [-1, 1] using tanh
        x = tf.tanh(inputs) if self.tanh_x else inputs

        x = tf.math.divide(
            tf.math.subtract(
                # (batch_size, 1, self.input_dim),
                tf.expand_dims(x, axis=1),
                # (1, output_dim, input_dim)
                self.translation
            ),
            # (1, output_dim, input_dim)
            tf.math.exp(self.scale)
        )
    
        return tf.einsum(
            'boi,oi->bo', 
            self.get_wavelets(x), 
            self.wavelet_weights,
            optimize='auto'
        )

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

    def get_config(self):
        config = super().get_config()
        config.update({
            "scale_init": self.scale_init,
            "translation_init": self.translation_init,
        })
        return config