from abc import abstractmethod
import tensorflow as tf
import numpy as np

from arnold.layers.core.kan_base import KANBase

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="WaveletBase")
class WaveletBase(KANBase):
    r"""
    Abstract base class for Kolmogorov-Arnold Network layer using wavelets.

    :ivar scale: The (learnable) non-zero, positive scale/dilation parameter for the wavelet transform. Internally stored as log(scale).
    :vartype scale: tf.Tensor

    :ivar translation: The (learnable) translation parameter fro the wavelet transform.
    :vartype translation: tf.Tensor
    
    :ivar wavelet_weights: The learnable wavelet coefficients.
    :vartype wavelet_weights: tf.Tensor
    """

    def __init__(self, 
                 *args,
                 scale_init: float | None = None, scale_trainable=True, 
                 translation_init: float | None = None, translation_trainable=True, 
                 **kwargs):
        r"""
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

        x_scaled = tf.math.divide(
            tf.math.subtract(
                # (batch_size, 1, self.input_dim),
                tf.expand_dims(x, axis=1),
                # (1, output_dim, input_dim)
                self.translation
            ),
            # (1, output_dim, input_dim)
            tf.math.exp(self.scale)
        )
    
        # shape (B,O,I)
        daughter_wavelets = self.mother_wavelet(x_scaled) / tf.math.sqrt(tf.math.abs(self.scale))

        return tf.einsum(
            'boi,oi->bo', 
            # self.get_wavelets(x_scaled), 
            daughter_wavelets,
            self.wavelet_weights,
            optimize='auto'
        )
    
    @abstractmethod
    def mother_wavelet(self, x):
        r"""
        Computes the daughter wavelets for given input tensor `x`.


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


class Bump(WaveletBase):
    r"""
    Kolmogorov-Arnold Network layer using Bump wavelets :math:`\psi(x) = \mathbf{I}_{[\mu -\sigma, \mu+\sigma]}(x) e^{(1-\frac{1}{1-(x-\mu)^2 / \sigma^2})}`
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def mother_wavelet(self, x):
        # TODO: make eps a configurable parameter?
        eps = 1e-07

        x = tf.clip_by_value(x, -1.0+eps, 1.0-eps)
        return tf.exp(-1.0 / (1 - x**2))


class DerivativeOfGaussian(WaveletBase):
    r"""
    Kolmogorov-Arnold Network layer using DerivativeOfGaussian wavelets.
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def mother_wavelet(self, x):
        return -x * tf.exp(-0.5 * x**2)


class Meyer(WaveletBase):
    r"""
    Kolmogorov-Arnold Network layer using a Meyer wavelet :math:`\psi(x)=\frac{1}{3\pi} \F()`
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def mother_wavelet(self, x):

        def __nu(t):
            # See: https://de.mathworks.com/help/wavelet/ref/meyeraux.html
            return t**4 * (35 - 84*t + 70*t**2 - 20*t**3)
        
        def __meyer_aux(v):
            return tf.where(
                v <= 1/2,
                tf.ones_like(v),
                tf.where(
                    v >= 1,
                    tf.zeros_like(v),
                    tf.cos(np.pi / 2 * __nu(2 * v - 1))
                )
            )
        
        v = tf.abs(x)
        
        return tf.sin(np.pi * v) * __meyer_aux(v)


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
        :type omega_trainable: bool

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

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def mother_wavelet(self, x):
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


class Poisson(WaveletBase):
    r"""
    Kolmogorov-Arnold Network layer using Poisson wavelets :math:`{\psi (t)={\frac {1}{\pi }}{\frac {1-t^{2}}{(1+t^{2})^{2}}}}`
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def mother_wavelet(self, x):
        return (1 / np.pi) * tf.math.divide(
            tf.math.subtract(1.0, tf.square(x)),
            tf.square(tf.math.add(1.0, tf.square(x)))
        )


class Ricker(WaveletBase):
    r"""
    Kolmogorov-Arnold Network layer using Ricker (Mexican hat) wavelet :math:`\psi (t)={\frac {2}{{\sqrt {3\sigma }}\pi ^{1/4}}}\left(1-\left({\frac {t}{\sigma }}\right)^{2}\right)e^{-{\frac {t^{2}}{2\sigma ^{2}}}}`.
    """

    def __init__(self, 
                 *args,
                 sigma_init:float=1.0, 
                 sigma_trainable=True,
                 **kwargs):
        r"""
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

        :param sigma_init: Initial value for the sigma parameter of the Ricker wavelet. Defaults to 1.0.
        :type sigma_init: float

        :param sigma_trainable: Flag indicating whether sigma is a trainable parameter. Defaults to True
        :type sigma_trainable: bool

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
    
    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def mother_wavelet(self, x):
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


class Shannon(WaveletBase):
    r"""
    Kolmogorov-Arnold Network layer using Shannon wavelets :math:`\psi^{(Sha)}(t)=\mathop{\mathrm{sinc}} \left({\frac {t}{2}}\right)\cdot \cos \left({\frac {3\pi t}{2}}\right)`.
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def mother_wavelet(self, x):
        # See: https://mathworld.wolfram.com/SincFunction.html
        indices = tf.where(tf.equal(x, 0))
        x = tf.math.divide_no_nan(tf.math.sin(x), x)
        sinc_x = tf.tensor_scatter_nd_update(x, indices, tf.ones((tf.shape(indices)[0], )))
        
        # Applying a Hamming window to limit the infinite support of the sinc function
        window = tf.signal.hamming_window(tf.shape(x)[-1], periodic=False, dtype=x.dtype)
        
        # Shannon wavelet is the product of the sinc function and the window
        return (sinc_x * window)

