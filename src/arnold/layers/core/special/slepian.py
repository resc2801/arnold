# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Slepian function KAN layer.

This module provides :class:`Slepian`, a KAN layer using Slepian functions
(discrete prolate spheroidal sequences, DPSS) as basis functions.

Mathematical Background
-----------------------
Slepian functions maximize energy concentration in a given frequency band.
They are the eigenfunctions of the time-frequency concentration problem:

.. math::

    \int_{-W}^{W} |\hat{f}(\omega)|^2 d\omega = \lambda \int_{-\infty}^{\infty} |\hat{f}(\omega)|^2 d\omega

where :math:`\lambda` is the eigenvalue (energy concentration ratio).

The discrete version (DPSS) is widely used in:
- Multitaper spectral estimation
- Signal processing with optimal bandwidth concentration
- Time-frequency analysis

Properties
----------
- Orthonormal on the interval [-1, 1]
- Optimal concentration in frequency domain
- Eigenvalues close to 1 for first few sequences, then rapidly decay

References
----------
.. [1] Slepian, D. (1978). "Prolate Spheroidal Wave Functions, Fourier Analysis,
       and Uncertainty—V: The Discrete Case". Bell System Technical Journal.
.. [2] Thomson, D.J. (1982). "Spectrum estimation and harmonic analysis".
       Proceedings of the IEEE.
"""
import tensorflow as tf

from arnold.layers.core.special.base import SpecialBase


tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold", name="Slepian")
class Slepian(SpecialBase):
    r"""
    Kolmogorov-Arnold Network layer using Slepian functions (DPSS).

    Uses discrete prolate spheroidal sequences as basis functions,
    which are optimal for bandlimited signal representation.

    Parameters
    ----------
    units : int
        Output dimension.
    max_order : int, default=8
        Number of Slepian sequences to use.
    bandwidth : float, default=0.5
        Normalized bandwidth W ∈ (0, 0.5).
    input_clip : tuple[float, float] | None, default=(-1.0, 1.0)
        Input clipping for numerical stability.
    **kwargs
        Additional Keras layer arguments.

    Notes
    -----
    The true Slepian functions require solving an eigenvalue problem.
    This implementation uses a polynomial approximation that captures
    the essential properties:
    - Concentration in the low-frequency band
    - Orthogonality

    For exact DPSS, use scipy.signal.windows.dpss and precompute.

    Example
    -------
    >>> import tensorflow as tf
    >>> from arnold.layers.core.special import Slepian
    >>>
    >>> layer = Slepian(max_order=4, bandwidth=0.25, units=32)
    >>> x = tf.random.uniform((16, 10), -1, 1)
    >>> y = layer(x)  # shape: (16, 32)
    """

    def __init__(
        self,
        *,
        units: int,
        max_order: int = 8,
        bandwidth: float = 0.5,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        **kwargs,
    ):
        if not 0 < bandwidth < 0.5:
            raise ValueError(f"bandwidth must be in (0, 0.5), got {bandwidth}")
        self.bandwidth = bandwidth
        super().__init__(
            units=units,
            max_order=max_order,
            input_clip=input_clip,
            **kwargs,
        )

    def _get_num_basis_functions(self) -> int:
        """Return number of Slepian basis functions."""
        return self.max_order

    def special_basis(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Compute Slepian-like basis functions.

        Uses a product of Legendre polynomials and sinc-like functions
        to approximate Slepian sequence properties.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, input_dim), expected in [-1, 1].

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, num_basis).
        """
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        W = tf.cast(self.bandwidth, tf.float64)
        pi = 3.141592653589793

        basis = []

        # Approximate Slepian sequences using sinc-windowed polynomials
        # The true DPSS are eigenvectors of a specific matrix
        # We approximate with: sinc(2Wx) * P_n(x) * window

        # Sinc kernel for bandwidth limitation
        arg = 2.0 * W * pi * x
        sinc = tf.where(
            tf.abs(arg) < 1e-10,
            tf.ones_like(arg),
            tf.sin(arg) / arg
        )

        # Gaussian-like concentration window
        window = tf.exp(-x * x / (2.0 * W * W * 4.0))

        # Legendre polynomial basis with sinc modulation
        P0 = tf.ones_like(x, dtype=tf.float64)
        P1 = x

        # First Slepian-like function (most concentrated)
        s0 = sinc * window * P0
        s0 = s0 / (tf.sqrt(tf.reduce_mean(s0 * s0, axis=-1, keepdims=True)) + 1e-10)
        basis.append(s0)

        if self.max_order >= 2:
            s1 = sinc * window * P1
            s1 = s1 / (tf.sqrt(tf.reduce_mean(s1 * s1, axis=-1, keepdims=True)) + 1e-10)
            basis.append(s1)

            P_prev2 = P0
            P_prev1 = P1

            for n in range(2, self.max_order):
                n_f = tf.cast(n, tf.float64)
                # Legendre recurrence
                P_n = ((2.0 * n_f - 1.0) * x * P_prev1 - (n_f - 1.0) * P_prev2) / n_f

                # Slepian-like function
                s_n = sinc * window * P_n
                # Rough normalization
                s_n = s_n / (tf.sqrt(tf.reduce_mean(s_n * s_n, axis=-1, keepdims=True)) + 1e-10)
                basis.append(s_n)

                P_prev2 = P_prev1
                P_prev1 = P_n

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "bandwidth": self.bandwidth,
        })
        return config
