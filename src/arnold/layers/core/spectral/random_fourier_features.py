# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Random Fourier Features KAN layer.

This module provides :class:`RandomFourierFeatures`, a KAN layer using random
Fourier feature expansions to approximate RBF kernels.

Mathematical Background
-----------------------
Random Fourier Features (Rahimi & Recht, 2007) provide a scalable approximation
to shift-invariant kernels. For the Gaussian RBF kernel:

.. math::

    k(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)

the feature map is:

.. math::

    \phi(x) = \sqrt{\frac{2}{D}} \left[ \cos(\omega_1^T x + b_1), \ldots, \cos(\omega_D^T x + b_D) \right]

where :math:`\omega_i \sim \mathcal{N}(0, \sigma^{-2} I)` and :math:`b_i \sim \text{Uniform}(0, 2\pi)`.

Key Properties:
- **Kernel approximation**: :math:`\phi(x)^T \phi(y) \approx k(x, y)`
- **Scalability**: :math:`O(D)` vs :math:`O(n^2)` for explicit kernel
- **Trainable**: Frequencies can be fine-tuned during training
- **Universal**: Approximates any shift-invariant kernel

References
----------
.. [1] Rahimi, A., & Recht, B. (2007). Random Features for Large-Scale
       Kernel Machines. NeurIPS.
.. [2] Liu, Z., et al. (2021). Random Features for Kernel Approximation:
       A Survey. arXiv:2004.11154.

Example
-------
>>> from arnold.layers.core.spectral import RandomFourierFeatures
>>> import tensorflow as tf
>>> layer = RandomFourierFeatures(units=32, num_features=128, kernel_scale=1.0)
>>> x = tf.random.normal((16, 10))  # batch=16, features=10
>>> y = layer(x)  # shape: (16, 32)
"""
import math

import tensorflow as tf

from arnold.layers.core.spectral.base import SpectralBase

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold", name="RandomFourierFeatures")
class RandomFourierFeatures(SpectralBase):
    r"""
    Random Fourier Features KAN layer — approximates RBF kernel.

    Computes random Fourier feature expansions:

    .. math::

        \phi_d(x) = \sqrt{\frac{2}{D}} \cos(\omega_d^T x + b_d)

    where :math:`\omega_d \sim \mathcal{N}(0, \gamma^2 I)` and
    :math:`b_d \sim \text{Uniform}(0, 2\pi)`.

    Parameters
    ----------
    units : int
        Output dimensionality.
    num_features : int, default=64
        Number of random features :math:`D`. Higher = better kernel approximation.
    kernel_scale : float, default=1.0
        RBF kernel bandwidth :math:`\gamma = 1/\sigma`. Larger = narrower kernel.
    trainable_frequencies : bool, default=False
        If True, the random frequencies :math:`\omega` become trainable parameters.
    trainable_phases : bool, default=False
        If True, the phase offsets :math:`b` become trainable parameters.
    input_clip : tuple[float, float] | None, default=None
        Optional input clamping range.
    use_bias : bool, default=True
        Whether to add a bias term to the output.
    activation : str or callable, default=None
        Activation function to apply after the layer.
    kernel_regularizer : regularizer, default=None
        Regularizer for the output projection weights.
    seed : int | None, default=None
        Random seed for reproducible frequency initialization.

    Attributes
    ----------
    _omega : tf.Variable
        Random frequency matrix of shape ``(input_dim, num_features)``.
    _phase : tf.Variable
        Random phase offsets of shape ``(num_features,)``.

    Notes
    -----
    **Kernel Approximation Quality:**
    - Error decreases as :math:`O(1/\sqrt{D})` with num_features
    - For 1% relative error, typically need :math:`D \geq 1000`
    - Trainable frequencies can improve approximation for specific data

    **Relationship to RBF Networks:**
    - RFF-KAN is a scalable alternative to RBF-KAN
    - Avoids explicit center selection and :math:`O(n^2)` kernel computation
    - Can approximate same functions with learned linear combinations

    **Initialization:**
    - Frequencies sampled from :math:`\mathcal{N}(0, \gamma^2 I)`
    - Phases sampled uniformly from :math:`[0, 2\pi]`
    - Both can be made trainable for task-specific adaptation

    Example
    -------
    >>> import tensorflow as tf
    >>> from arnold.layers.core.spectral import RandomFourierFeatures
    >>>
    >>> # Approximate Gaussian kernel for SVM-like features
    >>> layer = RandomFourierFeatures(
    ...     units=64,
    ...     num_features=256,
    ...     kernel_scale=0.5,  # wider kernel
    ...     trainable_frequencies=True,
    ... )
    >>> x = tf.random.uniform((32, 8), -1, 1)
    >>> y = layer(x)  # shape: (32, 64)
    >>>
    >>> # The features approximate: k(x, y) ≈ φ(x)ᵀφ(y)
    """

    def __init__(
        self,
        *,
        units: int,
        num_features: int = 64,
        kernel_scale: float = 1.0,
        trainable_frequencies: bool = False,
        trainable_phases: bool = False,
        input_clip: tuple[float, float] | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        if num_features < 1:
            raise ValueError(f"num_features must be >= 1, got {num_features}")
        if kernel_scale <= 0:
            raise ValueError(f"kernel_scale must be > 0, got {kernel_scale}")

        self.num_features = num_features
        self.kernel_scale = float(kernel_scale)
        self.trainable_frequencies = trainable_frequencies
        self.trainable_phases = trainable_phases
        self.seed = seed

        self._omega = None
        self._phase = None

        super().__init__(
            units=units,
            num_frequencies=num_features,
            input_clip=input_clip,
            **kwargs,
        )

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        # Initialize random frequencies: ω ~ N(0, γ²I)
        # For RBF kernel k(x,y) = exp(-γ²||x-y||²/2), we sample ω ~ N(0, γ²I)
        rng = tf.random.Generator.from_seed(self.seed) if self.seed is not None else tf.random.get_global_generator()
        
        omega_init = rng.normal(
            shape=(input_dim, self.num_features),
            stddev=self.kernel_scale,
        )
        
        self._omega = self.add_weight(
            name="omega",
            shape=(input_dim, self.num_features),
            initializer=tfk.initializers.Constant(omega_init.numpy()),
            trainable=self.trainable_frequencies,
        )

        # Initialize random phases: b ~ Uniform(0, 2π)
        phase_init = rng.uniform(
            shape=(self.num_features,),
            minval=0.0,
            maxval=2.0 * math.pi,
        )
        
        self._phase = self.add_weight(
            name="phase",
            shape=(self.num_features,),
            initializer=tfk.initializers.Constant(phase_init.numpy()),
            trainable=self.trainable_phases,
        )

        super().build(input_shape)

    def _get_num_basis_functions(self) -> int:
        """Return the number of random Fourier features."""
        return self.num_features

    def spectral_basis(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Compute random Fourier feature basis at input locations.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape ``(..., input_dim)``.

        Returns
        -------
        tf.Tensor
            Basis values of shape ``(..., input_dim, num_features)``.

        Notes
        -----
        The feature map is:

        .. math::

            \phi_d(x_i) = \sqrt{\frac{2}{D}} \cos(\omega_d^T x_i + b_d)

        We compute this per input dimension to maintain the KAN structure.
        """
        # x shape: (..., input_dim)
        # omega shape: (input_dim, num_features)
        # phase shape: (num_features,)

        omega = tf.cast(self._omega, x.dtype)
        phase = tf.cast(self._phase, x.dtype)

        # Compute per-dimension projection: x_i * omega[i, :] + phase
        # This maintains the univariate KAN structure
        # x[..., i] * omega[i, d] + phase[d] for each input dimension i
        
        # Expand x: (..., input_dim, 1)
        x_expanded = x[..., tf.newaxis]
        
        # Broadcast: x[..., i, 1] * omega[i, d] -> (..., input_dim, num_features)
        projection = x_expanded * omega + phase  # (..., input_dim, num_features)

        # Apply cosine and normalize
        # Normalization factor sqrt(2/D) ensures E[φ(x)ᵀφ(y)] ≈ k(x, y)
        norm_factor = tf.cast(tf.sqrt(2.0 / self.num_features), x.dtype)
        basis = norm_factor * tf.cos(projection)

        return basis

    def get_config(self):
        config = super().get_config()
        # Remove num_frequencies to avoid duplicate argument (it's derived from num_features)
        config.pop("num_frequencies", None)
        config.update({
            "num_features": self.num_features,
            "kernel_scale": self.kernel_scale,
            "trainable_frequencies": self.trainable_frequencies,
            "trainable_phases": self.trainable_phases,
            "seed": self.seed,
        })
        return config

    @property
    def frequencies(self) -> tf.Tensor:
        """Current random frequency matrix."""
        return self._omega

    @property
    def phases(self) -> tf.Tensor:
        """Current phase offset vector."""
        return self._phase
