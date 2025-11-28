# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
RBF Base Class
==============

Abstract base class for Radial Basis Function KAN layers.
"""

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
