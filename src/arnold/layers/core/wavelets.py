## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
from abc import abstractmethod
from typing import Literal

import numpy as np
import tensorflow as tf

from arnold.layers.core.kan_base import KANBase
from arnold.utils.compilation import kan_function
from arnold.utils.constants import PARAM_EPS


tfk = tf.keras
tfkl = tfk.layers
kan_fn = kan_function()  # @kan_fn = @tf.function(jit_compile=True)


# ============================================================================
# Filter Coefficients for Discrete Wavelets (Daubechies, Symlet, Coiflet)
# ============================================================================
# Reference: PyWavelets / NIST Digital Library of Mathematical Functions
# These are the low-pass filter coefficients h[n] for the scaling function.
# The wavelet filter g[n] is derived via the QMF relation.

DAUBECHIES_COEFFICIENTS = {
    1: np.array([1.0, 1.0]) / np.sqrt(2),  # db1 = Haar
    2: np.array([
        0.4829629131445341,
        0.8365163037378079,
        0.2241438680420134,
        -0.1294095225512604,
    ]),
    3: np.array([
        0.3326705529500826,
        0.8068915093110925,
        0.4598775021184915,
        -0.1350110200102546,
        -0.0854412738820267,
        0.0352262918857095,
    ]),
    4: np.array([
        0.2303778133088965,
        0.7148465705529156,
        0.6308807679298589,
        -0.0279837694168599,
        -0.1870348117190931,
        0.0308413818355607,
        0.0328830116668852,
        -0.0105974017850690,
    ]),
    5: np.array([
        0.1601023979741930,
        0.6038292697971896,
        0.7243085284377729,
        0.1384281459013204,
        -0.2422948870663824,
        -0.0322448695846381,
        0.0775714938400459,
        -0.0062414902127983,
        -0.0125807519990820,
        0.0033357252854738,
    ]),
    6: np.array([
        0.1115407433501095,
        0.4946238903984533,
        0.7511339080210959,
        0.3152503517091982,
        -0.2262646939654400,
        -0.1297668675672625,
        0.0975016055873225,
        0.0275228655303053,
        -0.0315820393174862,
        0.0005538422011614,
        0.0047772575109455,
        -0.0010773010853085,
    ]),
    7: np.array([
        0.0778520540850037,
        0.3965393194819173,
        0.7291320908462351,
        0.4697822874051931,
        -0.1439060039285293,
        -0.2240361849938749,
        0.0713092192668312,
        0.0806126091510820,
        -0.0380299369350125,
        -0.0165745416306664,
        0.0125509985560993,
        0.0004295779729214,
        -0.0018016407040474,
        0.0003537137999745,
    ]),
    8: np.array([
        0.0544158422431049,
        0.3128715909143031,
        0.6756307362972904,
        0.5853546836541907,
        -0.0158291052563823,
        -0.2840155429615702,
        0.0004724845739124,
        0.1287474266204837,
        -0.0173693010018083,
        -0.0440882539307952,
        0.0139810279173995,
        0.0087460940474061,
        -0.0048703529934518,
        -0.0003917403733770,
        0.0006754494064506,
        -0.0001174767841248,
    ]),
    10: np.array([
        0.0266700579005473,
        0.1881768000776347,
        0.5272011889317255,
        0.6884590394536250,
        0.2811723436606485,
        -0.2498464243273153,
        -0.1959462743773399,
        0.1273693403357890,
        0.0930573646035802,
        -0.0713941471663697,
        -0.0294575368218480,
        0.0332126740593703,
        0.0036065535669880,
        -0.0107331754833036,
        0.0013953517470688,
        0.0019924052951930,
        -0.0006858566949566,
        -0.0001164668551285,
        0.0000935886703202,
        -0.0000132642028945,
    ]),
}

SYMLET_COEFFICIENTS = {
    2: np.array([
        -0.1294095225512604,
        0.2241438680420134,
        0.8365163037378079,
        0.4829629131445341,
    ]),
    3: np.array([
        0.0352262918857095,
        -0.0854412738820267,
        -0.1350110200102546,
        0.4598775021184915,
        0.8068915093110925,
        0.3326705529500826,
    ]),
    4: np.array([
        -0.0757657147893407,
        -0.0296355276459541,
        0.4976186676324578,
        0.8037387518052163,
        0.2978577956055422,
        -0.0992195435769354,
        -0.0126039672622612,
        0.0322231006040713,
    ]),
    5: np.array([
        0.0273330683451645,
        0.0295194909260734,
        -0.0391342493025834,
        0.1993975339773936,
        0.7234076904038076,
        0.6339789634569490,
        0.0166021057644243,
        -0.1753280899081075,
        -0.0211018340249298,
        0.0195388827353869,
    ]),
    6: np.array([
        0.0154041093273377,
        0.0034907120843304,
        -0.1179901111484105,
        -0.0483117425859981,
        0.4910559419276396,
        0.7876411410287941,
        0.3379294217282401,
        -0.0726375227866000,
        -0.0210602925126954,
        0.0447249017707482,
        0.0017677118643983,
        -0.0078007083247650,
    ]),
    7: np.array([
        0.0102681767084968,
        0.0040102448717033,
        -0.1078082377036168,
        -0.1400472404427030,
        0.2886296317509833,
        0.7677643170045710,
        0.5361019170907720,
        0.0174412550871099,
        -0.0495528349370410,
        0.0678926935015971,
        0.0305155131659062,
        -0.0126363034031526,
        -0.0010473848889657,
        0.0026818145681164,
    ]),
    8: np.array([
        -0.0033824159513594,
        -0.0005421323316355,
        0.0316950878103452,
        0.0076074873252848,
        -0.1432942383510542,
        -0.0612733590679088,
        0.4813596512592012,
        0.7771857516997478,
        0.3644418948359564,
        -0.0519458381078751,
        -0.0272190299168137,
        0.0491371796734768,
        0.0038087520140601,
        -0.0149522583367926,
        -0.0003029205145516,
        0.0018899503329007,
    ]),
}

COIFLET_COEFFICIENTS = {
    1: np.array([
        -0.0156557285289848,
        -0.0727326213410511,
        0.3848648565381134,
        0.8525720416423900,
        0.3378976709511590,
        -0.0727322757411889,
    ]),
    2: np.array([
        -0.0007205494453679,
        -0.0018232088707116,
        0.0056114348194211,
        0.0236801719464464,
        -0.0594344186467388,
        -0.0764885990786692,
        0.4170051844236707,
        0.8127236354493977,
        0.3861100668229939,
        -0.0673725547222826,
        -0.0414649367819558,
        0.0163873364635998,
    ]),
    3: np.array([
        -0.0000345997728362,
        -0.0000709833031381,
        0.0004662169601129,
        0.0011175187708906,
        -0.0025745176887502,
        -0.0090079761366615,
        0.0158805448636158,
        0.0345550275730615,
        -0.0823019271068856,
        -0.0717998216193117,
        0.4284834763776168,
        0.7937772226256169,
        0.4051769024096150,
        -0.0611233900026726,
        -0.0657719112818552,
        0.0234526961418362,
        0.0077825964273254,
        -0.0037935128644910,
    ]),
    4: np.array([
        -0.0000017849850031,
        -0.0000032596802369,
        0.0000312298758654,
        0.0000623390344610,
        -0.0002599745524878,
        -0.0005890207562444,
        0.0012665619292991,
        0.0037514361572790,
        -0.0056582866866115,
        -0.0152117315279485,
        0.0250822618448678,
        0.0393344271233433,
        -0.0962204420340021,
        -0.0666274742634348,
        0.4343860564915321,
        0.7822389309206135,
        0.4153084070304910,
        -0.0560773133167630,
        -0.0812666996808907,
        0.0266823001560570,
        0.0160689439647787,
        -0.0073461663276432,
        -0.0016294920126020,
        0.0008923136685824,
    ]),
    5: np.array([
        -0.0000000951765727,
        -0.0000001674428858,
        0.0000020637618516,
        0.0000037346551755,
        -0.0000213150268122,
        -0.0000413404322768,
        0.0001405411497166,
        0.0003022595818445,
        -0.0006381313431115,
        -0.0016628637021860,
        0.0024333732129107,
        0.0067641854487565,
        -0.0091642311634348,
        -0.0197617789446276,
        0.0326835742705106,
        0.0412892087544753,
        -0.1055742087143175,
        -0.0620359639693546,
        0.4379916262173834,
        0.7742896037334738,
        0.4215662067346898,
        -0.0520431631816557,
        -0.0919200105692549,
        0.0281680289738655,
        0.0234081567882734,
        -0.0101132506418967,
        -0.0041003308476738,
        0.0021060292512300,
        0.0003604674532865,
        -0.0002580451227311,
    ]),
}


@tfk.utils.register_keras_serializable(package="arnold", name="WaveletBase")
class WaveletBase(KANBase):
    r"""
    Abstract base class for Kolmogorov-Arnold Network layers using wavelets.

    This layer computes:

    .. math::

        y_j = \sum_{i=1}^{d_{\text{in}}} w_{i,j} \cdot \frac{1}{\sqrt{s_j}} \, 
              \psi\!\left(\frac{x_i - t_j}{s_j}\right) + b_j

    where :math:`\psi` is the mother wavelet, :math:`s_j > 0` is the learnable scale,
    :math:`t_j` is the learnable translation, and :math:`w_{i,j}` are learnable weights.

    The :math:`1/\sqrt{s}` normalization preserves the :math:`L^2` norm of the wavelet
    across scales, ensuring energy consistency.

    Parameters are stored as unconstrained logits and mapped to valid values
    via :func:`tf.nn.softplus`. Scaling uses a small :math:`\varepsilon` floor
    to avoid division by zero; translations remain unconstrained.

    Notes
    -----
    Common mother wavelets include:

    - **Ricker (Mexican hat)**: :math:`\psi(t) = \frac{2}{\sqrt{3\sigma}\pi^{1/4}}(1 - (t/\sigma)^2) e^{-t^2/(2\sigma^2)}`
    - **Morlet**: :math:`\psi(t) = e^{i\omega_0 t} e^{-t^2/2}`
    - **Derivative of Gaussian (DOG)**: :math:`\psi(t) = -t \, e^{-t^2/2}`
    - **Meyer**: Band-limited wavelet with smooth frequency cutoffs

    See Also
    --------
    Ricker, Morelet, DerivativeOfGaussian, Meyer, Shannon, Bump, Poisson
    """

    def __init__(
        self,
        units: int,
        scale_init: float | None = None,
        scale_trainable: bool = True,
        translation_init: float | None = None,
        translation_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        units : int
            Output dimensionality.
        scale_init : float | None
            Initial positive scale value; defaults to HeNormal when None.
        scale_trainable : bool
            Whether scale is trainable.
        translation_init : float | None
            Initial translation; defaults to HeNormal when None.
        translation_trainable : bool
            Whether translation is trainable.
        input_clip : tuple[float, float], optional
            Optional input clamp.
        **kwargs :
            Forwarded to :class:`KANBase`.
        """
        super().__init__(units=units, input_clip=input_clip, **kwargs)

        if (scale_init is not None) and (scale_init <= 0):
            raise ValueError("Non-zero, positive value for the initial wavelet scale parameter required!")

        self.scale_init = scale_init
        self.translation_init = translation_init
        self.scale_trainable = scale_trainable
        self.translation_trainable = translation_trainable
        self.scale = None
        self.translation = None
        self.wavelet_weights = None

    def build(self, input_shape):
        super().build(input_shape)

        scale_init = (
            tfk.initializers.Constant(value=tf.math.log(self.scale_init))
            if self.scale_init
            else tfk.initializers.HeNormal()
        )
        translation_init = (
            tfk.initializers.Constant(value=self.translation_init)
            if self.translation_init is not None
            else tfk.initializers.HeNormal()
        )

        self.scale = self.add_weight(
            shape=(1, self.output_dim, self.input_dim),
            initializer=scale_init,
            name="scale_logits",
            trainable=self.scale_trainable,
        )

        self.translation = self.add_weight(
            shape=(1, self.output_dim, self.input_dim),
            initializer=translation_init,
            name="translation",
            trainable=self.translation_trainable,
        )

        self.wavelet_weights = self.add_weight(
            shape=(self.output_dim, self.input_dim),
            initializer=tfk.initializers.HeUniform(),
            regularizer=self.kernel_regularizer,
            name="wavelet_weights",
            trainable=True,
        )

    def call(self, inputs):
        x = self._preprocess_inputs(inputs)
        original_dtype = x.dtype
        leading_shape = tf.shape(x)[:-1]

        # Cast to effective compute dtype for mixed-precision support
        compute_dtype = self.effective_compute_dtype
        if x.dtype != compute_dtype:
            x = tf.cast(x, compute_dtype)

        scale = tf.nn.softplus(self.scale) + tf.cast(PARAM_EPS, compute_dtype)
        x_scaled = tf.math.divide(
            tf.expand_dims(x, axis=1) - self.translation,
            scale,
        )

        daughter_wavelets = self.mother_wavelet(x_scaled) / tf.math.sqrt(scale)

        y_flat = tf.einsum(
            "boi,oi->bo",
            daughter_wavelets,
            self.wavelet_weights,
            optimize="auto",
        )

        # Cast back to original dtype
        if y_flat.dtype != original_dtype:
            y_flat = tf.cast(y_flat, original_dtype)

        y_flat = self._apply_activation_and_bias(y_flat)
        return tf.reshape(y_flat, tf.concat([leading_shape, [self.output_dim]], axis=0))

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
            f"Layer {self.__class__.__name__} does not have a `get_wavelets()` method implemented."
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "scale_init": self.scale_init,
                "translation_init": self.translation_init,
                "scale_trainable": self.scale_trainable,
                "translation_trainable": self.translation_trainable,
            }
        )
        return config


class Bump(WaveletBase):
    r"""
    Kolmogorov-Arnold Network layer using Bump wavelets :math:`\psi(x) = \mathbf{I}_{[-1,1]}(x)\, e^{(1-\frac{1}{1-x^2})}`.
    """

    def __init__(self, *args, units: int, input_clip=None, **kwargs):
        """
        Parameters
        ----------
        units : int
            Output dimensionality.
        input_clip : tuple[float, float] | None
            Optional clamp of inputs before wavelet evaluation.
        """
        super().__init__(*args, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def mother_wavelet(self, x):
        eps = 1e-07
        x = tf.clip_by_value(x, -1.0 + eps, 1.0 - eps)
        return tf.exp(-1.0 / (1 - x**2))


class DerivativeOfGaussian(WaveletBase):
    r"""
    Kolmogorov-Arnold Network layer using the first derivative of a Gaussian wavelet.
    """

    def __init__(self, *args, units: int, input_clip=None, **kwargs):
        super().__init__(*args, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def mother_wavelet(self, x):
        return -x * tf.exp(-0.5 * x**2)


class Meyer(WaveletBase):
    r"""
    Kolmogorov-Arnold Network layer using a Meyer wavelet.
    """

    def __init__(self, *args, units: int, input_clip=None, **kwargs):
        super().__init__(*args, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def mother_wavelet(self, x):
        def __nu(t):
            # See: https://de.mathworks.com/help/wavelet/ref/meyeraux.html
            return t**4 * (35 - 84 * t + 70 * t**2 - 20 * t**3)

        def __meyer_aux(v):
            return tf.where(
                v <= 1 / 2, tf.ones_like(v), tf.where(v >= 1, tf.zeros_like(v), tf.cos(np.pi / 2 * __nu(2 * v - 1)))
            )

        v = tf.abs(x)

        return tf.sin(np.pi * v) * __meyer_aux(v)


class Morelet(WaveletBase):
    r"""
    Kolmogorov-Arnold Network layer using Morelet (Morlet) wavelets.
    """

    def __init__(
        self,
        *,
        units: int,
        omega_init: float = 5.0,
        omega_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        units : int
            Output dimensionality.
        omega_init : float
            Initial central frequency :math:`\omega_0` (must be positive).
        omega_trainable : bool
            Whether :math:`\omega_0` is trainable.
        input_clip : tuple[float, float], optional
            Optional input clamp.
        """
        super().__init__(units=units, input_clip=input_clip, **kwargs)

        self.omega_init = omega_init
        self.omega_trainable = omega_trainable
        self.omega0 = None

    def build(self, input_shape):
        super().build(input_shape)
        self.omega0 = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.omega_init),
            name="central_frequency",
            trainable=self.omega_trainable,
        )

    @kan_fn
    def mother_wavelet(self, x):
        real = tf.cos(self.omega0 * x)
        envelope = tf.exp(-0.5 * x**2)
        return envelope * real

    def get_config(self):
        config = super().get_config()
        config.update({"omega": self.omega_init, "omega_trainable": self.omega_trainable})
        return config


class Poisson(WaveletBase):
    r"""
    Kolmogorov-Arnold Network layer using Poisson wavelets :math:`{\psi (t)={\frac {1}{\pi }}{\frac {1-t^{2}}{(1+t^{2})^{2}}}}`
    """

    def __init__(self, *args, units: int, input_clip=None, **kwargs):
        super().__init__(*args, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def mother_wavelet(self, x):
        return (1 / np.pi) * tf.math.divide(
            tf.math.subtract(1.0, tf.square(x)), tf.square(tf.math.add(1.0, tf.square(x)))
        )


class Ricker(WaveletBase):
    r"""
    Kolmogorov-Arnold Network layer using the Ricker (Mexican hat) wavelet.

    The Ricker wavelet is the negative normalized second derivative of a Gaussian:

    .. math::

        \psi(t) = \frac{2}{\sqrt{3\sigma}\,\pi^{1/4}}
                  \left(1 - \left(\frac{t}{\sigma}\right)^2\right)
                  \exp\!\left(-\frac{t^2}{2\sigma^2}\right)

    where :math:`\sigma > 0` controls the wavelet width.

    Properties:

    - **Admissibility**: Zero mean (:math:`\int \psi(t) dt = 0`)
    - **Localization**: Compactly supported in frequency, well-localized in time
    - **Zero crossings**: At :math:`t = \pm\sigma`
    - **Peak**: Positive maximum at :math:`t = 0`

    Notes
    -----
    The Ricker wavelet is commonly used in seismic analysis and edge detection.
    It is particularly effective for detecting features at multiple scales when
    combined with learnable scale and translation parameters from ``WaveletBase``.

    See Also
    --------
    DerivativeOfGaussian : First derivative of Gaussian (odd symmetry)
    Morelet : Complex wavelet with oscillatory behavior
    """

    def __init__(
        self,
        *,
        units: int,
        sigma_init: float = 1.0,
        sigma_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        units : int
            Output dimensionality.
        sigma_init : float
            Initial standard deviation :math:`\sigma` (must be positive).
        sigma_trainable : bool
            Whether :math:`\sigma` is trainable.
        input_clip : tuple[float, float], optional
            Optional input clamp.
        """
        super().__init__(units=units, input_clip=input_clip, **kwargs)

        self.sigma_init = sigma_init
        self.sigma_trainable = sigma_trainable
        self.sigma = None

    def build(self, input_shape):
        super().build(input_shape)
        self.sigma = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.sigma_init),
            name="standard_deviation",
            trainable=self.sigma_trainable,
        )

    @kan_fn
    def mother_wavelet(self, x):
        sigma = tf.nn.softplus(self.sigma) + tf.cast(1e-6, x.dtype)
        term1 = 1.0 - tf.square(x / sigma)
        term2 = tf.exp(-0.5 * tf.square(x / sigma))
        return (2 / (tf.math.sqrt(3.0) * np.pi**0.25)) * term1 * term2

    def get_config(self):
        config = super().get_config()
        config.update({"sigma": self.sigma_init, "sigma_trainable": self.sigma_trainable})
        return config


class Shannon(WaveletBase):
    r"""
    Kolmogorov-Arnold Network layer using Shannon wavelets.

    The Shannon (sinc) wavelet is defined as:

    .. math::
        \psi(t) = \mathrm{sinc}(t) \cdot w(t)

    where :math:`\mathrm{sinc}(t) = \sin(\pi t) / (\pi t)` (normalized sinc)
    and :math:`w(t)` is a Hamming window to limit the infinite support.

    The wavelet is normalized to have unit L2 energy.

    Stability: The Hamming window is precomputed in ``build()`` to avoid
    dynamic shape issues in XLA. The sinc function handles the singularity
    at zero via ``tf.where``.
    """

    def __init__(self, *args, units: int, input_clip=None, **kwargs):
        super().__init__(*args, units=units, input_clip=input_clip, **kwargs)
        self._hamming_window = None
        self._normalization = None

    def build(self, input_shape):
        super().build(input_shape)
        # Precompute Hamming window with static shape (input_dim)
        # to avoid tf.signal.hamming_window inside jitted function
        dtype = tf.dtypes.as_dtype(self.compute_dtype)
        self._hamming_window = tf.signal.hamming_window(
            self.input_dim, periodic=False, dtype=dtype
        )
        # Precompute L2 normalization factor for unit energy
        # ||window||_2 for energy normalization
        self._normalization = tf.sqrt(tf.reduce_sum(tf.square(self._hamming_window)))

    @kan_fn
    def mother_wavelet(self, x):
        # Normalized sinc: sinc(t) = sin(πt)/(πt)
        # Handle singularity at x=0 where sinc(0) = 1
        pi_x = np.pi * x
        sinc_x = tf.where(
            tf.abs(x) < 1e-8,
            tf.ones_like(x),
            tf.math.sin(pi_x) / pi_x
        )

        # Apply precomputed Hamming window to limit infinite support
        window = tf.cast(self._hamming_window, x.dtype)
        norm = tf.cast(self._normalization, x.dtype)

        # Shannon wavelet: sinc * window, normalized for unit L2 energy
        return (sinc_x * window) / (norm + 1e-8)

    def get_config(self):
        return super().get_config()


# ============================================================================
# Filter-Bank Wavelets (Haar, Daubechies, Symlet, Coiflet)
# ============================================================================


class Haar(WaveletBase):
    r"""
    Kolmogorov-Arnold Network layer using the Haar wavelet.

    The Haar wavelet is the simplest orthogonal wavelet, defined as:

    .. math::

        \psi(t) = \begin{cases}
            1 & 0 \leq t < 1/2 \\
            -1 & 1/2 \leq t < 1 \\
            0 & \text{otherwise}
        \end{cases}

    The scaling function (father wavelet) is:

    .. math::

        \phi(t) = \begin{cases}
            1 & 0 \leq t < 1 \\
            0 & \text{otherwise}
        \end{cases}

    Properties
    ----------
    - **Compact support**: :math:`[0, 1]`
    - **Orthogonal**: Forms an orthonormal basis of :math:`L^2(\mathbb{R})`
    - **Piecewise constant**: Discontinuous, optimal for detecting edges
    - **Vanishing moments**: 1 (detects constant signals)

    Notes
    -----
    The Haar wavelet is equivalent to db1 (Daubechies order 1). It is
    particularly effective for edge detection and discontinuity localization.

    The implementation uses a smooth approximation via sigmoid functions
    to enable gradient-based learning while preserving the essential
    step-function character.

    See Also
    --------
    Daubechies : Generalized orthogonal wavelets with more vanishing moments
    """

    def __init__(
        self,
        *,
        units: int,
        sharpness: float = 20.0,
        sharpness_trainable: bool = False,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        units : int
            Output dimensionality.
        sharpness : float
            Controls the steepness of the sigmoid approximation.
            Higher values approach the true step function.
        sharpness_trainable : bool
            Whether sharpness is trainable.
        input_clip : tuple[float, float], optional
            Optional input clamp.
        """
        super().__init__(units=units, input_clip=input_clip, **kwargs)
        self.sharpness_init = sharpness
        self.sharpness_trainable = sharpness_trainable
        self._sharpness = None

    def build(self, input_shape):
        super().build(input_shape)
        self._sharpness = self.add_weight(
            shape=(),
            initializer=tfk.initializers.Constant(self.sharpness_init),
            name="sharpness",
            trainable=self.sharpness_trainable,
        )

    @kan_fn
    def mother_wavelet(self, x):
        # Smooth approximation to Haar wavelet using sigmoids
        # ψ(t) ≈ σ(k*t) - 2*σ(k*(t-0.5)) + σ(k*(t-1))
        # where k is sharpness and σ is sigmoid
        k = tf.nn.softplus(self._sharpness) + 1.0
        
        # Step function approximations
        step_0 = tf.sigmoid(k * x)           # step at 0
        step_half = tf.sigmoid(k * (x - 0.5))  # step at 0.5
        step_1 = tf.sigmoid(k * (x - 1.0))   # step at 1
        
        # Haar wavelet: 1 on [0, 0.5), -1 on [0.5, 1), 0 elsewhere
        # = step(x) - 2*step(x-0.5) + step(x-1)
        return step_0 - 2.0 * step_half + step_1

    def get_config(self):
        config = super().get_config()
        config.update({
            "sharpness": self.sharpness_init,
            "sharpness_trainable": self.sharpness_trainable,
        })
        return config


class Daubechies(WaveletBase):
    r"""
    Kolmogorov-Arnold Network layer using Daubechies wavelets.

    Daubechies wavelets are a family of orthogonal wavelets characterized
    by a maximal number of vanishing moments for a given support width.

    The wavelet is defined via its filter coefficients :math:`h[n]`:

    .. math::

        \phi(t) = \sqrt{2} \sum_{n=0}^{2N-1} h[n] \phi(2t - n)

    where :math:`N` is the order (number of vanishing moments).

    Properties
    ----------
    - **Compact support**: :math:`[0, 2N-1]`
    - **Orthogonal**: Forms an orthonormal basis
    - **Vanishing moments**: :math:`N` (detects polynomials up to degree :math:`N-1`)
    - **Regularity**: Increases with order

    Parameters
    ----------
    order : int
        Order of the Daubechies wavelet (1-10). db1 = Haar.

    Notes
    -----
    Available orders: 1, 2, 3, 4, 5, 6, 7, 8, 10

    The implementation uses a Gaussian-mixture approximation of the wavelet
    function, which is differentiable and XLA-compatible while preserving
    the essential frequency characteristics of the Daubechies family.

    See Also
    --------
    Haar : Simplest case (db1)
    Symlet : Near-symmetric variant
    Coiflet : Variant with symmetric scaling function
    """

    def __init__(
        self,
        *,
        units: int,
        order: Literal[1, 2, 3, 4, 5, 6, 7, 8, 10] = 4,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        units : int
            Output dimensionality.
        order : int
            Daubechies order (1-10). db1 = Haar, db4 is most common.
        input_clip : tuple[float, float], optional
            Optional input clamp.
        """
        if order not in DAUBECHIES_COEFFICIENTS:
            available = sorted(DAUBECHIES_COEFFICIENTS.keys())
            raise ValueError(f"Daubechies order must be in {available}, got {order}")
        
        super().__init__(units=units, input_clip=input_clip, **kwargs)
        self.order = order
        self._g = None
        self._centers = None

    def build(self, input_shape):
        super().build(input_shape)
        # Store filter coefficients as non-trainable constants
        h = DAUBECHIES_COEFFICIENTS[self.order].astype(np.float32)
        # Wavelet filter via QMF relation: g[n] = (-1)^n * h[N-1-n]
        g = np.array([(-1) ** n * h[len(h) - 1 - n] for n in range(len(h))])
        self._g = tf.constant(g, dtype=self.compute_dtype, name="wavelet_filter")
        # Centers for Gaussian mixture (at filter coefficient positions)
        self._centers = tf.constant(
            np.arange(len(h), dtype=np.float32), 
            dtype=self.compute_dtype, 
            name="centers"
        )
        self._support = float(len(h) - 1)

    @kan_fn
    def mother_wavelet(self, x):
        # Vectorized Gaussian-mixture approximation of Daubechies wavelet
        # Each filter coefficient contributes a Gaussian centered at its position
        g = tf.cast(self._g, x.dtype)
        centers = tf.cast(self._centers, x.dtype)
        support = tf.cast(self._support, x.dtype)
        
        # Normalize x to wavelet support [0, support]
        t = x * support
        
        # Compute all contributions at once using broadcasting
        # t shape: (..., input_dim)
        # centers shape: (filter_len,)
        # We want: t[..., :, None] - centers[None, ...]
        t_expanded = tf.expand_dims(t, axis=-1)  # (..., input_dim, 1)
        shifts = t_expanded - centers  # (..., input_dim, filter_len)
        
        # Gaussian localization around each filter point
        # Width proportional to filter length for smoothness
        sigma = 0.5 + 0.1 * support
        gaussians = tf.exp(-0.5 * tf.square(shifts / sigma))  # (..., input_dim, filter_len)
        
        # Weight by filter coefficients and sum
        weighted = gaussians * g  # (..., input_dim, filter_len)
        psi = tf.reduce_sum(weighted, axis=-1)  # (..., input_dim)
        
        # Normalize
        return psi / (tf.sqrt(support) + 1e-8)

    def get_config(self):
        config = super().get_config()
        config.update({
            "order": self.order,
        })
        return config


class Symlet(WaveletBase):
    r"""
    Kolmogorov-Arnold Network layer using Symlet wavelets.

    Symlets are near-symmetric modifications of Daubechies wavelets.
    They have the same number of vanishing moments as Daubechies but
    with improved symmetry properties.

    Properties
    ----------
    - **Compact support**: Same as Daubechies of same order
    - **Near-symmetry**: More symmetric than Daubechies
    - **Vanishing moments**: Same as Daubechies of same order
    - **Better phase response**: Important for signal processing

    Parameters
    ----------
    order : int
        Order of the Symlet wavelet (2-8).

    Notes
    -----
    Symlets are often preferred over Daubechies when phase distortion
    matters, such as in signal reconstruction applications.

    See Also
    --------
    Daubechies : Original orthogonal wavelets
    Coiflet : Variant with symmetric scaling function
    """

    def __init__(
        self,
        *,
        units: int,
        order: Literal[2, 3, 4, 5, 6, 7, 8] = 4,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        units : int
            Output dimensionality.
        order : int
            Symlet order (2-8). Higher orders have more vanishing moments.
        input_clip : tuple[float, float], optional
            Optional input clamp.
        """
        if order not in SYMLET_COEFFICIENTS:
            available = sorted(SYMLET_COEFFICIENTS.keys())
            raise ValueError(f"Symlet order must be in {available}, got {order}")
        
        super().__init__(units=units, input_clip=input_clip, **kwargs)
        self.order = order
        self._g = None
        self._centers = None

    def build(self, input_shape):
        super().build(input_shape)
        h = SYMLET_COEFFICIENTS[self.order].astype(np.float32)
        g = np.array([(-1) ** n * h[len(h) - 1 - n] for n in range(len(h))])
        self._g = tf.constant(g, dtype=self.compute_dtype, name="wavelet_filter")
        self._centers = tf.constant(
            np.arange(len(h), dtype=np.float32),
            dtype=self.compute_dtype,
            name="centers"
        )
        self._support = float(len(h) - 1)

    @kan_fn
    def mother_wavelet(self, x):
        g = tf.cast(self._g, x.dtype)
        centers = tf.cast(self._centers, x.dtype)
        support = tf.cast(self._support, x.dtype)
        
        t = x * support
        t_expanded = tf.expand_dims(t, axis=-1)
        shifts = t_expanded - centers
        
        sigma = 0.5 + 0.1 * support
        gaussians = tf.exp(-0.5 * tf.square(shifts / sigma))
        weighted = gaussians * g
        psi = tf.reduce_sum(weighted, axis=-1)
        
        return psi / (tf.sqrt(support) + 1e-8)

    def get_config(self):
        config = super().get_config()
        config.update({
            "order": self.order,
        })
        return config


class Coiflet(WaveletBase):
    r"""
    Kolmogorov-Arnold Network layer using Coiflet wavelets.

    Coiflets are designed to have vanishing moments for both the
    wavelet and the scaling function, resulting in near-symmetric
    wavelets with excellent approximation properties.

    Properties
    ----------
    - **Compact support**: :math:`[0, 6N-1]` for order :math:`N`
    - **Symmetric**: Nearly symmetric scaling and wavelet functions
    - **Double vanishing moments**: Both :math:`\psi` and :math:`\phi` have :math:`2N` moments
    - **Excellent reconstruction**: Minimal distortion

    Parameters
    ----------
    order : int
        Order of the Coiflet wavelet (1-5).

    Notes
    -----
    Coiflets are named after Ronald Coifman. They are particularly
    useful when both the signal and its approximation need to be
    analyzed simultaneously.

    See Also
    --------
    Daubechies : Original orthogonal wavelets
    Symlet : Near-symmetric Daubechies variant
    """

    def __init__(
        self,
        *,
        units: int,
        order: Literal[1, 2, 3, 4, 5] = 2,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        units : int
            Output dimensionality.
        order : int
            Coiflet order (1-5). Higher orders have more vanishing moments.
        input_clip : tuple[float, float], optional
            Optional input clamp.
        """
        if order not in COIFLET_COEFFICIENTS:
            available = sorted(COIFLET_COEFFICIENTS.keys())
            raise ValueError(f"Coiflet order must be in {available}, got {order}")
        
        super().__init__(units=units, input_clip=input_clip, **kwargs)
        self.order = order
        self._g = None
        self._centers = None

    def build(self, input_shape):
        super().build(input_shape)
        h = COIFLET_COEFFICIENTS[self.order].astype(np.float32)
        g = np.array([(-1) ** n * h[len(h) - 1 - n] for n in range(len(h))])
        self._g = tf.constant(g, dtype=self.compute_dtype, name="wavelet_filter")
        self._centers = tf.constant(
            np.arange(len(h), dtype=np.float32),
            dtype=self.compute_dtype,
            name="centers"
        )
        self._support = float(len(h) - 1)

    @kan_fn
    def mother_wavelet(self, x):
        g = tf.cast(self._g, x.dtype)
        centers = tf.cast(self._centers, x.dtype)
        support = tf.cast(self._support, x.dtype)
        
        t = x * support
        t_expanded = tf.expand_dims(t, axis=-1)
        shifts = t_expanded - centers
        
        sigma = 0.5 + 0.1 * support
        gaussians = tf.exp(-0.5 * tf.square(shifts / sigma))
        weighted = gaussians * g
        psi = tf.reduce_sum(weighted, axis=-1)
        
        return psi / (tf.sqrt(support) + 1e-8)

    def get_config(self):
        config = super().get_config()
        config.update({
            "order": self.order,
        })
        return config
