# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Fourier/trigonometric basis KAN layer.

This module provides :class:`FourierKAN`, a KAN layer using trigonometric
basis functions for learning periodic patterns.

Mathematical Background
-----------------------
The Fourier basis represents functions as linear combinations of sinusoids:

.. math::

    f(x) = a_0 + \sum_{n=1}^{N} \left[ a_n \cos(n\omega x) + b_n \sin(n\omega x) \right]

This layer learns the coefficients :math:`a_n, b_n` while optionally also
learning the base frequency :math:`\omega`.

Key Properties:
- **Orthogonality**: Basis functions are orthogonal over :math:`[0, 2\pi/\omega]`
- **Periodicity**: Natural for periodic patterns (time series, angular data)
- **Spectral analysis**: Frequency components are interpretable
- **Universal approximation**: Dense in :math:`L^2` for continuous periodic functions

Example
-------
>>> from arnold.layers.core.spectral import FourierKAN
>>> import tensorflow as tf
>>> layer = FourierKAN(units=32, degree=8)
>>> x = tf.random.normal((16, 10))  # batch=16, features=10
>>> y = layer(x)  # shape: (16, 32)
"""

import tensorflow as tf

from arnold.layers.core.spectral.base import SpectralBase


tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold", name="FourierKAN")
class FourierKAN(SpectralBase):
    r"""
    Fourier/trigonometric basis KAN layer.

    Computes learnable linear combinations of Fourier basis functions:

    .. math::

        \phi_0(x) &= 1 \\
        \phi_{2k-1}(x) &= \cos(k \omega x), \quad k = 1, \ldots, n \\
        \phi_{2k}(x) &= \sin(k \omega x), \quad k = 1, \ldots, n

    Total number of basis functions: :math:`2n + 1` where :math:`n` is the degree.

    Parameters
    ----------
    units : int
        Output dimensionality.
    degree : int, default=8
        Maximum frequency order :math:`n`. Determines the number of harmonics.
    frequency : float, default=1.0
        Base frequency :math:`\omega`. For data on :math:`[-\pi, \pi]`, use 1.0.
        For data on :math:`[-1, 1]`, use :math:`\pi`.
    learnable_frequency : bool, default=False
        If True, the base frequency :math:`\omega` becomes a trainable parameter.
    input_clip : tuple[float, float] | None, default=None
        Optional input clamping range before Fourier evaluation.
    use_bias : bool, default=True
        Whether to add a bias term to the output.
    activation : str or callable, default=None
        Activation function to apply after the layer.
    kernel_regularizer : regularizer, default=None
        Regularizer for the Fourier coefficients.

    Attributes
    ----------
    _omega : tf.Variable
        The (potentially trainable) base frequency parameter.

    Notes
    -----
    **Numerical Stability:**
    - Fourier basis is unconditionally stable for any input range
    - No overflow concerns for high degrees (unlike polynomials)
    - Gradient magnitudes are bounded by the degree

    **Interpretability:**
    - Coefficient magnitudes indicate frequency importance
    - Can extract dominant frequencies from trained model
    - Natural for spectral analysis of learned functions

    **Domain Considerations:**
    - For inputs on :math:`[-1, 1]`: set ``frequency=π``
    - For inputs on :math:`[-\pi, \pi]`: set ``frequency=1.0``
    - For arbitrary domains: scale inputs or adjust frequency

    Example
    -------
    >>> import tensorflow as tf
    >>> from arnold.layers.core.spectral import FourierKAN
    >>>
    >>> # Learn periodic patterns in time series
    >>> layer = FourierKAN(
    ...     units=64,
    ...     degree=16,
    ...     frequency=2*3.14159,  # period = 1
    ...     learnable_frequency=True,
    ... )
    >>> x = tf.random.uniform((32, 8), -1, 1)
    >>> y = layer(x)  # shape: (32, 64)
    """

    def __init__(
        self,
        *,
        units: int,
        degree: int = 8,
        frequency: float = 1.0,
        learnable_frequency: bool = False,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        if frequency <= 0:
            raise ValueError(f"frequency must be > 0, got {frequency}")

        self.degree = degree
        self.frequency_init = float(frequency)
        self.learnable_frequency = learnable_frequency
        self._omega = None

        super().__init__(
            units=units,
            num_frequencies=degree,
            input_clip=input_clip,
            **kwargs,
        )

    def build(self, input_shape):
        # Create frequency parameter before calling super().build()
        self._omega = self.add_weight(
            name="omega",
            shape=(),
            initializer=tfk.initializers.Constant(self.frequency_init),
            trainable=self.learnable_frequency,
        )

        super().build(input_shape)

    def _get_num_basis_functions(self) -> int:
        """Return 2*degree + 1 basis functions: {1, cos, sin, cos2, sin2, ...}."""
        return 2 * self.degree + 1

    def spectral_basis(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Compute Fourier basis functions at input locations.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape ``(..., input_dim)``.

        Returns
        -------
        tf.Tensor
            Basis values of shape ``(..., input_dim, 2*degree+1)``.

        Notes
        -----
        The basis is ordered as:

        .. math::

            [\underbrace{1}_{\text{DC}},
             \underbrace{\cos(\omega x), \sin(\omega x)}_{\text{1st harmonic}},
             \underbrace{\cos(2\omega x), \sin(2\omega x)}_{\text{2nd harmonic}},
             \ldots]
        """
        # x shape: (..., input_dim)
        omega = tf.cast(self._omega, x.dtype)

        # Frequency multipliers: [1, 2, 3, ..., degree]
        n = tf.range(1, self.degree + 1, dtype=x.dtype)  # shape: (degree,)

        # Compute n * omega * x for all harmonics
        # x[..., i] * n[k] * omega -> [..., input_dim, degree]
        x_expanded = x[..., tf.newaxis]  # (..., input_dim, 1)
        phases = x_expanded * n * omega  # (..., input_dim, degree)

        # Compute cos and sin
        cos_terms = tf.cos(phases)  # (..., input_dim, degree)
        sin_terms = tf.sin(phases)  # (..., input_dim, degree)

        # Interleave: [cos(ωx), sin(ωx), cos(2ωx), sin(2ωx), ...]
        # Shape: (..., input_dim, 2*degree)
        cos_sin = tf.stack([cos_terms, sin_terms], axis=-1)  # (..., input_dim, degree, 2)
        cos_sin = tf.reshape(
            cos_sin,
            tf.concat([tf.shape(x), [2 * self.degree]], axis=0),
        )  # (..., input_dim, 2*degree)

        # Prepend DC component (constant 1)
        ones = tf.ones_like(x)[..., tf.newaxis]  # (..., input_dim, 1)
        basis = tf.concat([ones, cos_sin], axis=-1)  # (..., input_dim, 2*degree+1)

        return basis

    def get_config(self):
        config = super().get_config()
        # Remove num_frequencies to avoid duplicate argument (it's derived from degree)
        config.pop("num_frequencies", None)
        config.update({
            "degree": self.degree,
            "frequency": self.frequency_init,
            "learnable_frequency": self.learnable_frequency,
        })
        return config

    @property
    def omega(self) -> tf.Tensor:
        """Current base frequency value."""
        return self._omega
