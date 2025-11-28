# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Spherical Harmonics KAN layer.

This module provides :class:`SphericalHarmonics`, a KAN layer using spherical
harmonic basis functions for rotationally equivariant representations.

Mathematical Background
-----------------------
Spherical harmonics :math:`Y_l^m(\theta, \phi)` are eigenfunctions of the
Laplace-Beltrami operator on the 2-sphere :math:`S^2`:

.. math::

    Y_l^m(\theta, \phi) = N_l^m P_l^{|m|}(\cos\theta) e^{im\phi}

where:
- :math:`l \geq 0` is the degree
- :math:`-l \leq m \leq l` is the order
- :math:`P_l^m` are associated Legendre polynomials
- :math:`N_l^m` is the normalization constant

Properties:
- **Orthonormality**: :math:`\int Y_l^m Y_{l'}^{m'*} d\Omega = \delta_{ll'}\delta_{mm'}`
- **Completeness**: Any :math:`L^2(S^2)` function can be expanded in spherical harmonics
- **Rotational equivariance**: Transform predictably under SO(3) rotations

For 1D inputs, this layer uses the real spherical harmonics evaluated at
fixed angles, effectively computing associated Legendre polynomial combinations.

Applications:
- 3D point cloud processing
- Molecular property prediction
- SE(3)-equivariant neural networks
- Spherical CNNs

References
----------
.. [1] Weisstein, E.W. "Spherical Harmonic." MathWorld.
.. [2] Cohen, T.S., et al. (2018). "Spherical CNNs". ICLR.
.. [3] Thomas, N., et al. (2018). "Tensor Field Networks". arXiv:1802.08219.
"""

import tensorflow as tf

from arnold.layers.core.geometric.base import GeometricBase


tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold", name="SphericalHarmonics")
class SphericalHarmonics(GeometricBase):
    r"""
    Kolmogorov-Arnold Network layer using spherical harmonic basis functions.

    For 1D inputs, this layer computes real spherical harmonics-like basis functions
    based on Legendre polynomials. The basis consists of:

    .. math::

        \phi_l(x) = P_l(x), \quad l = 0, 1, \ldots, l_{\max}

    where :math:`P_l(x)` are Legendre polynomials, which are the :math:`m=0`
    spherical harmonics (zonal harmonics).

    For rotationally equivariant applications, use this with 3D angular inputs
    (θ, φ) or with specialized input preprocessing.

    Parameters
    ----------
    max_degree : int, default=8
        Maximum degree :math:`l_{\max}`. Total basis functions: :math:`l_{\max} + 1`.
    units : int
        Output dimension.
    use_real : bool, default=True
        If True, use real spherical harmonics (cosine/sine combinations).
        If False, use complex exponentials (not implemented for 1D).
    input_clip : tuple[float, float] | None, default=(-1.0, 1.0)
        Input clipping range. Legendre polynomials are defined on [-1, 1].
    use_bias : bool, default=True
        Whether to add a bias term.
    activation : str or callable, default=None
        Activation function.
    kernel_regularizer : regularizer, default=None
        Regularizer for harmonic coefficients.

    Notes
    -----
    **Connection to Spherical Harmonics:**
    For 1D inputs x = cos(θ), the zonal harmonics are:

    .. math::

        Y_l^0(\theta) = \sqrt{\frac{2l+1}{4\pi}} P_l(\cos\theta)

    This layer uses unnormalized Legendre polynomials for simplicity.

    **Numerical Stability:**
    Uses three-term recurrence for Legendre polynomials:

    .. math::

        (l+1) P_{l+1}(x) = (2l+1) x P_l(x) - l P_{l-1}(x)

    This is stable for :math:`|x| \leq 1` and moderate degrees.

    Example
    -------
    >>> import tensorflow as tf
    >>> from arnold.layers.core.geometric import SphericalHarmonics
    >>>
    >>> layer = SphericalHarmonics(max_degree=8, units=32)
    >>> x = tf.random.uniform((16, 10), -1, 1)  # cos(θ) in [-1, 1]
    >>> y = layer(x)  # shape: (16, 32)
    """

    def __init__(
        self,
        *,
        units: int,
        max_degree: int = 8,
        use_real: bool = True,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        **kwargs,
    ):
        self.use_real = use_real
        super().__init__(
            units=units,
            max_degree=max_degree,
            input_clip=input_clip,
            **kwargs,
        )

    def _get_num_basis_functions(self) -> int:
        """Return number of basis functions: max_degree + 1 Legendre polynomials."""
        return self.max_degree + 1

    def geometric_basis(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Compute Legendre polynomial basis (zonal spherical harmonics).

        Uses the three-term recurrence relation:

        .. math::

            (l+1) P_{l+1}(x) = (2l+1) x P_l(x) - l P_{l-1}(x)

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, input_dim), expected in [-1, 1].

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, max_degree+1).
        """
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)

        # P_0(x) = 1
        P0 = tf.ones_like(x, dtype=tf.float64)
        basis = [P0]

        if self.max_degree >= 1:
            # P_1(x) = x
            P1 = x
            basis.append(P1)

            P_prev2 = P0  # P_{l-2}
            P_prev1 = P1  # P_{l-1}

            for l in range(2, self.max_degree + 1):
                # (l) P_l(x) = (2l-1) x P_{l-1}(x) - (l-1) P_{l-2}(x)
                l_f = tf.cast(l, tf.float64)
                P_l = ((2.0 * l_f - 1.0) * x * P_prev1 - (l_f - 1.0) * P_prev2) / l_f
                basis.append(P_l)
                P_prev2 = P_prev1
                P_prev1 = P_l

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"use_real": self.use_real})
        return config
