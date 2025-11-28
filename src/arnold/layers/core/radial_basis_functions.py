## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
from abc import abstractmethod

import tensorflow as tf

from arnold.layers.core.kan_base import KANBase
from arnold.utils.compilation import kan_function
from arnold.utils.constants import PARAM_EPS


tfk = tf.keras
tfkl = tfk.layers
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="RBFBase")
class RBFBase(KANBase):
    r"""
    Abstract base class for Kolmogorov-Arnold Network layers using radial basis functions.

    This layer computes:

    .. math::

        y_j = \sum_{i=1}^{d_{\text{in}}} \sum_{k=1}^{K} w_{i,k,j} \, \phi\!\left(\frac{\|x_i - \mu_k\|}{h}\right) + b_j

    where :math:`\phi(r)` is the radial kernel, :math:`\mu_k` are equidistant grid centers
    in :math:`[\text{grid\_min}, \text{grid\_max}]`, :math:`h` is the grid spacing,
    and :math:`w_{i,k,j}` are learnable weights.

    Parameters are kept in logits and transformed with ``softplus`` to stay strictly
    positive, avoiding zero-divisions or kernel collapse.

    Notes
    -----
    Common RBF kernels include:

    - **Gaussian**: :math:`\phi(r) = \exp(-(\varepsilon r)^2)`
    - **Multiquadric**: :math:`\phi(r) = \sqrt{1 + (\varepsilon r)^2}`
    - **Inverse Multiquadric**: :math:`\phi(r) = 1 / \sqrt{1 + (\varepsilon r)^2}`
    - **Thin Plate Spline**: :math:`\phi(r) = r^2 \ln(r)`

    See Also
    --------
    GaussianRBF, MultiquadricRBF, InverseMultiquadricRBF, ThinPlateSplineRBF
    """

    def __init__(
        self,
        *,
        units: int,
        grid_min: float = 0.0,
        grid_max: float = 1.0,
        num_grids: int = 8,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        units : int, optional
            Output dimensionality. Alias for ``output_dim`` in :class:`KANBase`.
        grid_min : float
            Lower bound for the radial grid (default 0).
        grid_max : float
            Upper bound for the radial grid (default 1).
        num_grids : int
            Number of equidistant grid points. Must be >= 2.
        input_clip : tuple[float, float], optional
            Optional input clamp.
        **kwargs :
            Forwarded to :class:`KANBase`.
        """
        super().__init__(units=units, input_clip=input_clip, **kwargs)
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids

        self.grid = None
        self.kernel_weights = None

    def build(self, input_shape):
        super().build(input_shape)

        # Use effective_compute_dtype to respect mixed-precision policy
        self.grid = tf.linspace(
            tf.cast(self.grid_min, self.effective_compute_dtype),
            tf.cast(self.grid_max, self.effective_compute_dtype),
            self.num_grids,
        )
        if self.num_grids < 2:
            raise ValueError("`num_grids` must be at least 2 to form a radial grid.")
        if self.grid_max <= self.grid_min:
            raise ValueError("`grid_max` must be greater than `grid_min`.")

        self.kernel_weights = self.add_weight(
            shape=(self.input_dim, self.num_grids, self.output_dim),
            initializer=tfk.initializers.RandomNormal(),
            regularizer=self.kernel_regularizer,
            trainable=True,
            name="kernel_weights",
        )
        self.built = True

    @kan_fn
    def radii(self, x):
        r"""
        Computes :math:`\lVert x  - x_i \rVert` for equidistant :math:`x_i \in [\text{grid_min}, \text{grid_max}]`

        :param x: Data points to compute the radii with.
        :type inputs: tf.Tensor

        :returns: :math:`\lVert x  - x_i \rVert`
        :rtype: tf.Tensor
        """
        grid = tf.cast(self.grid, x.dtype)
        spacing = (self.grid_max - self.grid_min) / tf.cast(self.num_grids - 1, x.dtype)
        spacing = tf.maximum(spacing, tf.constant(PARAM_EPS, dtype=x.dtype))
        return tf.math.abs((x - grid) / spacing)

    def call(self, inputs):
        x = self._preprocess_inputs(inputs)
        original_dtype = x.dtype
        leading_shape = tf.shape(x)[:-1]

        # Cast to effective compute dtype for mixed-precision support
        compute_dtype = self.effective_compute_dtype
        if x.dtype != compute_dtype:
            x = tf.cast(x, compute_dtype)

        x_flat = tf.reshape(x, (-1, self.input_dim, 1))

        kernels = self.get_kernels(self.radii(x_flat))
        y_flat = tf.einsum(
            "bid,ido->bo",
            kernels,
            self.kernel_weights,
            optimize="auto",
        )

        # Cast back to original dtype
        if y_flat.dtype != original_dtype:
            y_flat = tf.cast(y_flat, original_dtype)

        y_flat = self._apply_activation_and_bias(y_flat)
        return tf.reshape(y_flat, tf.concat([leading_shape, [self.output_dim]], axis=0))

    @abstractmethod
    def get_kernels(self, r):
        r"""
        Evaluates the radial basis kernels for given :math:`r = \lVert x - x_{i} \rVert`.

        :param r: Data to compute the basis with.
        :type r: tf.Tensor

        :returns: Evaluated radial kernels
        :rtype: tf.Tensor
        """
        raise NotImplementedError(
            f"Layer {self.__class__.__name__} does not have a `get_kernels()` method implemented."
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "grid_min": self.grid_min,
                "grid_max": self.grid_max,
                "num_grids": self.num_grids,
            }
        )
        return config

    @staticmethod
    def _positive_from_logits(logits: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
        """Convert free logits to a strictly positive parameter with floor ``eps``."""
        return tf.nn.softplus(logits) + tf.cast(eps, logits.dtype)


class ExponentialRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the Exponential radial basis function

    :math:`\phi(r) = e^{ - \frac{r}{\sigma}}, \quad \sigma \in \mathbb{R}, r = \lVert x - x_{i} \rVert`

    Stability: :math:`\sigma` is stored as logits and mapped to
    ``softplus(sigma_logits) + eps`` to keep it positive and well-conditioned.
    """

    def __init__(
        self,
        *,
        units: int,
        sigma_init: float | None = None,
        sigma_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        units : int
            Output dimensionality.
        sigma_init : float | None
            Initial value for shape parameter :math:`\sigma`; defaults to RandomNormal when None.
        sigma_trainable : bool
            Whether :math:`\sigma` is trainable.
        input_clip : tuple[float, float] | None
            Optional input clamp.
        **kwargs :
            Forwarded to :class:`RBFBase`.
        """
        super().__init__(units=units, input_clip=input_clip, **kwargs)
        self.sigma_init = sigma_init
        self.sigma_trainable = sigma_trainable
        self.sigma_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        initializer = (
            tfk.initializers.Constant(value=tf.math.log(self.sigma_init))
            if self.sigma_init
            else tfk.initializers.RandomNormal()
        )
        self.sigma_logits = self.add_weight(
            shape=(1, self.input_dim, 1),
            initializer=initializer,
            trainable=self.sigma_trainable,
            name="sigma_logits",
        )

    @kan_fn
    def get_kernels(self, r):
        sigma = self._positive_from_logits(self.sigma_logits)
        return tf.math.exp(-(r / sigma))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sigma_init": self.sigma_init,
                "sigma_trainable": self.sigma_trainable,
            }
        )
        return config


class PowerRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the Power radial basis function

    :math:`\phi(r) = r^{p}, \quad p \in \mathbb{R}, r = \lVert x - x_{i} \rVert`

    Stability: exponent :math:`p` is constrained to be positive via
    ``softplus``; inputs are floored to avoid ``0**p`` underflow.
    """

    def __init__(
        self,
        *,
        units: int,
        power_init: float | None = None,
        power_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        units : int
            Output dimensionality.
        power_init : float, optional
            Initial exponent :math:`p`; must be positive. Defaults to RandomNormal logits.
        power_trainable : bool
            Whether :math:`p` is trainable.
        input_clip : tuple[float, float], optional
            Optional input clamp.
        **kwargs :
            Forwarded to :class:`RBFBase`.
        """
        super().__init__(units=units, input_clip=input_clip, **kwargs)

        self.power_init = power_init
        self.power_trainable = power_trainable
        self.power_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        initializer = (
            tfk.initializers.Constant(value=tf.math.log(self.power_init))
            if self.power_init
            else tfk.initializers.RandomNormal()
        )
        self.power_logits = self.add_weight(
            shape=(1, self.input_dim, 1),
            initializer=initializer,
            trainable=self.power_trainable,
            name="power_logits",
        )

    @kan_fn
    def get_kernels(self, r):
        power = self._positive_from_logits(self.power_logits)
        r_safe = tf.maximum(r, tf.cast(1e-6, r.dtype))
        return tf.math.pow(r_safe, power)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "power_init": self.power_init,
                "power_trainable": self.power_trainable,
            }
        )
        return config


class ThinPlateSplineRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the Thin plate spline radial basis function

    :math:`\phi(r) = r^{2} \ln(r), \quad r = \lVert x - x_{i} \rVert`

    Stability: applies a small floor to :math:`r` before the logarithm to avoid
    ``log(0)`` while preserving gradients.
    """

    def __init__(self, *, units: int, input_clip=None, **kwargs):
        """
        Parameters
        ----------
        units : int
            Output dimensionality.
        input_clip : tuple[float, float] | None
            Optional input clamp.
        """
        super().__init__(units=units, input_clip=input_clip, **kwargs)

    @tf.function(
        autograph=True,
        jit_compile=True,
        reduce_retracing=True,
        experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
    )
    def get_kernels(self, r):
        r"""
        :math:`\phi(r) = r^{2} \ln(r)` with a small floor to avoid :math:`\log(0)`.
        """
        r_safe = tf.maximum(r, tf.cast(1e-6, r.dtype))
        return tf.square(r_safe) * tf.math.log(r_safe)


class GaussianRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the Gaussian radial basis function.

    The Gaussian RBF kernel is defined as:

    .. math::

        \phi(r) = \exp\!\left(-(\varepsilon r)^2\right), \quad r = \|x - \mu_k\|

    where :math:`\varepsilon > 0` is the shape parameter controlling the kernel width,
    and :math:`\mu_k` are the grid centers.

    The shape parameter :math:`\varepsilon` is stored in logits and transformed via
    ``softplus`` to ensure positivity and smooth gradient flow.

    Notes
    -----
    - Small :math:`\varepsilon`: wide, smooth kernels → global influence
    - Large :math:`\varepsilon`: narrow, peaked kernels → local influence

    For optimal interpolation, choose ``num_grids`` to roughly match the expected
    number of "features" in your input domain. The kernel width adapts during training.

    See Also
    --------
    MultiquadricRBF : Grows unboundedly, good for global approximation
    InverseMultiquadricRBF : Decays like Gaussian but with polynomial tails
    """

    def __init__(
        self,
        *,
        units: int,
        epsilon_init: float | None = None,
        epsilon_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        units : int
            Output dimensionality.
        epsilon_init : float, optional
            Initial positive shape parameter :math:`\epsilon`.
        epsilon_trainable : bool
            Whether :math:`\epsilon` is trainable.
        input_clip : tuple[float, float], optional
            Optional input clamp.
        **kwargs :
            Forwarded to :class:`RBFBase`.
        """
        super().__init__(units=units, input_clip=input_clip, **kwargs)

        self.epsilon_init = epsilon_init
        self.epsilon_trainable = epsilon_trainable
        self.epsilon_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        initializer = (
            tfk.initializers.Constant(value=tf.math.log(self.epsilon_init))
            if self.epsilon_init
            else tfk.initializers.RandomNormal()
        )
        self.epsilon_logits = self.add_weight(
            shape=(1, self.input_dim, 1),
            initializer=initializer,
            trainable=self.epsilon_trainable,
            name="epsilon_logits",
        )

    @tf.function(
        autograph=True,
        jit_compile=True,
        reduce_retracing=True,
        experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
    )
    def get_kernels(self, r):
        eps_param = self._positive_from_logits(self.epsilon_logits)
        return tf.exp(-((eps_param * r) ** 2))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "epsilon_init": self.epsilon_init,
                "epsilon_trainable": self.epsilon_trainable,
            }
        )
        return config


class CubicRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the Cubic radial basis function

    :math:`\phi(r) = r^{3}, \quad r = \lVert x - x_{i} \rVert`
    """

    def __init__(self, *, units: int, input_clip=None, **kwargs):
        super().__init__(units=units, input_clip=input_clip, **kwargs)

    @tf.function(
        autograph=True,
        jit_compile=True,
        reduce_retracing=True,
        experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
    )
    def get_kernels(self, r):
        return tf.math.pow(r, 3)


class LinearRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the Linear radial basis function

    :math:`\phi(r) = r, \quad r = \lVert x - x_{i} \rVert`
    """

    def __init__(self, *, units: int, input_clip=None, **kwargs):
        super().__init__(units=units, input_clip=input_clip, **kwargs)

    @tf.function(
        autograph=True,
        jit_compile=True,
        reduce_retracing=True,
        experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
    )
    def get_kernels(self, r):
        return r


class InverseQuadricRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using a inverse quadric radial basis function

    :math:`\phi(r) = \frac{1}(1+(\epsilon r)^{2}}, \quad r = \lVert x - x_{i} \rVert`

    Stability: :math:`\epsilon` is kept positive via ``softplus`` to prevent
    division by zero.
    """

    def __init__(
        self,
        *,
        units: int,
        epsilon_init: float | None = None,
        epsilon_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        units : int
            Output dimensionality.
        epsilon_init : float, optional
            Initial positive shape parameter :math:`\epsilon`.
        epsilon_trainable : bool
            Whether :math:`\epsilon` is trainable.
        input_clip : tuple[float, float], optional
            Optional input clamp.
        """
        super().__init__(units=units, input_clip=input_clip, **kwargs)

        self.epsilon_init = epsilon_init
        self.epsilon_trainable = epsilon_trainable
        self.epsilon_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        initializer = (
            tfk.initializers.Constant(value=tf.math.log(self.epsilon_init))
            if self.epsilon_init
            else tfk.initializers.RandomNormal()
        )
        self.epsilon_logits = self.add_weight(
            shape=(1, self.input_dim, 1),
            initializer=initializer,
            trainable=self.epsilon_trainable,
            name="epsilon_logits",
        )

    @tf.function(
        autograph=True,
        jit_compile=True,
        reduce_retracing=True,
        experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
    )
    def get_kernels(self, r):
        eps_param = self._positive_from_logits(self.epsilon_logits)
        return tf.math.divide(1.0, (1.0 + (eps_param * r) ** 2))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "epsilon_init": self.epsilon_init,
                "epsilon_trainable": self.epsilon_trainable,
            }
        )
        return config


class MultiquadricRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the Multiquadric radial basis function

    :math:`\phi(r) = \sqrt{1+(\epsilon r)^{2}}, \quad r = \lVert x - x_{i} \rVert`

    with shape-parameter tuning per input dimension.

    Stability: :math:`\epsilon` uses ``softplus`` to maintain positive curvature.
    """

    def __init__(
        self,
        *,
        units: int,
        epsilon_init: float | None = None,
        epsilon_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        units : int
            Output dimensionality.
        epsilon_init : float, optional
            Initial positive shape parameter :math:`\epsilon`.
        epsilon_trainable : bool
            Whether :math:`\epsilon` is trainable.
        input_clip : tuple[float, float], optional
            Optional input clamp.
        """
        super().__init__(units=units, input_clip=input_clip, **kwargs)

        self.epsilon_init = epsilon_init
        self.epsilon_trainable = epsilon_trainable
        self.epsilon_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        initializer = (
            tfk.initializers.Constant(value=tf.math.log(self.epsilon_init))
            if self.epsilon_init
            else tfk.initializers.RandomNormal()
        )
        self.epsilon_logits = self.add_weight(
            shape=(1, self.input_dim, 1),
            initializer=initializer,
            trainable=self.epsilon_trainable,
            name="epsilon_logits",
        )

    @tf.function(
        autograph=True,
        jit_compile=True,
        reduce_retracing=True,
        experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
    )
    def get_kernels(self, r):
        eps_param = self._positive_from_logits(self.epsilon_logits)
        return tf.math.sqrt(1.0 + (eps_param * r) ** 2)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "epsilon_init": self.epsilon_init,
                "epsilon_trainable": self.epsilon_trainable,
            }
        )
        return config


class InverseMultiQuadricRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the inverse multiquadric radial basis function

    :math:`\phi(r) = \frac{1}(\sqrt{1+(\epsilon r)^{2}}}, \quad r = \lVert x - x_{i} \rVert`

    with shape-parameter tuning per input dimension.

    Stability: :math:`\epsilon` uses ``softplus`` to avoid singularities.
    """

    def __init__(
        self,
        *,
        units: int,
        epsilon_init: float | None = None,
        epsilon_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        units : int
            Output dimensionality.
        epsilon_init : float, optional
            Initial positive shape parameter :math:`\epsilon`.
        epsilon_trainable : bool
            Whether :math:`\epsilon` is trainable.
        input_clip : tuple[float, float], optional
            Optional input clamp.
        """
        super().__init__(units=units, input_clip=input_clip, **kwargs)

        self.epsilon_init = epsilon_init
        self.epsilon_trainable = epsilon_trainable
        self.epsilon_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        initializer = (
            tfk.initializers.Constant(value=tf.math.log(self.epsilon_init))
            if self.epsilon_init
            else tfk.initializers.RandomNormal()
        )
        self.epsilon_logits = self.add_weight(
            shape=(1, self.input_dim, 1),
            initializer=initializer,
            trainable=self.epsilon_trainable,
            name="epsilon_logits",
        )

    @tf.function(
        autograph=True,
        jit_compile=True,
        reduce_retracing=True,
        experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
    )
    def get_kernels(self, r):
        eps_param = self._positive_from_logits(self.epsilon_logits)
        return tf.math.divide(1.0, tf.math.sqrt(1.0 + (eps_param * r) ** 2))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "epsilon_init": self.epsilon_init,
                "epsilon_trainable": self.epsilon_trainable,
            }
        )
        return config


class CauchyRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the Cauchy radial basis function

    :math:`\phi(r) = \frac{1}{1 + (\frac{r}{\sigma})^{2}}, \quad \sigma \in \mathbb{R}, r = \lVert x - x_{i} \rVert`

    Stability: :math:`\sigma` is kept positive via ``softplus`` to ensure
    bounded kernels.
    """

    def __init__(
        self,
        *,
        units: int,
        sigma_init: float | None = None,
        sigma_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        units : int
            Output dimensionality.
        sigma_init : float | None
            Initial value for shape parameter :math:`\sigma`; defaults to RandomNormal when None.
        sigma_trainable : bool
            Whether :math:`\sigma` is trainable.
        input_clip : tuple[float, float] | None
            Optional input clamp.
        **kwargs :
            Forwarded to :class:`RBFBase`.
        """
        super().__init__(units=units, input_clip=input_clip, **kwargs)

        self.sigma_init = sigma_init
        self.sigma_trainable = sigma_trainable
        self.sigma_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        initializer = (
            tfk.initializers.Constant(value=tf.math.log(self.sigma_init))
            if self.sigma_init
            else tfk.initializers.RandomNormal()
        )
        self.sigma_logits = self.add_weight(
            shape=(1, self.input_dim, 1),
            initializer=initializer,
            trainable=self.sigma_trainable,
            name="sigma_logits",
        )

    @tf.function(
        autograph=True,
        jit_compile=True,
        reduce_retracing=True,
        experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
    )
    def get_kernels(self, r):
        sigma = self._positive_from_logits(self.sigma_logits)
        return tf.math.divide(1.0, 1.0 + tf.math.square(tf.math.divide(r, sigma)))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sigma_init": self.sigma_init,
                "sigma_trainable": self.sigma_trainable,
            }
        )
        return config
