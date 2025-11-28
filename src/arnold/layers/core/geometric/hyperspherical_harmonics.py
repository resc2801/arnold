# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Hyperspherical Harmonics KAN layer.

This module provides :class:`HypersphericalHarmonics`, a KAN layer using
harmonics on n-dimensional spheres for rotationally equivariant representations.

Mathematical Background
-----------------------
Hyperspherical harmonics are the generalization of spherical harmonics to
the n-sphere :math:`S^{n-1}` embedded in :math:`\mathbb{R}^n`. They are
eigenfunctions of the Laplace-Beltrami operator on the hypersphere.

For the unit hypersphere in :math:`\mathbb{R}^n`, the harmonics of degree :math:`l`
are polynomials of degree :math:`l` restricted to the sphere, forming an
irreducible representation of SO(n).

The Gegenbauer (ultraspherical) polynomials :math:`C_l^{\alpha}(x)` with
:math:`\alpha = (n-2)/2` are the zonal hyperspherical harmonics — the
analog of Legendre polynomials for higher dimensions.

Properties:
- **Dimension**: Number of linearly independent degree-:math:`l` harmonics is
  :math:`\binom{n+l-1}{l} - \binom{n+l-3}{l-2}` for :math:`l \geq 2`
- **Orthogonality**: With respect to the surface measure on :math:`S^{n-1}`
- **Completeness**: Dense in :math:`L^2(S^{n-1})`

Applications:
- 3D point cloud processing (n=3, reduces to spherical harmonics)
- Molecular property prediction with SE(3) equivariance
- 4D rotations in computer vision (quaternion spaces)
- High-dimensional data on hyperspheres

References
----------
.. [1] Müller, C. (1966). "Spherical Harmonics". Springer.
.. [2] Higuchi, A. (1987). "Symmetric tensor spherical harmonics on the N-sphere".
       J. Math. Phys.
.. [3] Cohen, T.S., et al. (2019). "Gauge Equivariant Convolutional Networks".
       ICML.
"""
import math

import tensorflow as tf

from arnold.layers.core.geometric.base import GeometricBase

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold", name="HypersphericalHarmonics")
class HypersphericalHarmonics(GeometricBase):
    r"""
    Kolmogorov-Arnold Network layer using hyperspherical harmonic basis functions.

    For 1D inputs, this layer computes Gegenbauer (ultraspherical) polynomials
    :math:`C_l^{\alpha}(x)` which are the zonal hyperspherical harmonics for
    :math:`S^{n-1}` with :math:`\alpha = (n-2)/2`.

    Special cases:
    - **n=2** (:math:`\alpha=0`): Chebyshev polynomials of the first kind
    - **n=3** (:math:`\alpha=1/2`): Legendre polynomials (spherical harmonics)
    - **n=4** (:math:`\alpha=1`): Chebyshev polynomials of the second kind

    Parameters
    ----------
    units : int
        Output dimension.
    dimension : int, default=3
        Dimension n of the ambient space :math:`\mathbb{R}^n`. The hypersphere
        is :math:`S^{n-1}`. Use 3 for standard spherical harmonics.
    max_degree : int, default=8
        Maximum degree :math:`l_{\max}`. Total basis functions: :math:`l_{\max} + 1`.
    input_clip : tuple[float, float] | None, default=(-1.0, 1.0)
        Input clipping range. Gegenbauer polynomials are defined on [-1, 1].
    use_bias : bool, default=True
        Whether to add a bias term.
    activation : str or callable, default=None
        Activation function.
    kernel_regularizer : regularizer, default=None
        Regularizer for harmonic coefficients.

    Attributes
    ----------
    alpha : float
        Gegenbauer parameter :math:`\alpha = (n-2)/2`.

    Notes
    -----
    **Gegenbauer Polynomial Recurrence:**

    .. math::

        C_l^{\alpha}(x) = \frac{1}{l} \left[ 2(l + \alpha - 1) x C_{l-1}^{\alpha}(x)
            - (l + 2\alpha - 2) C_{l-2}^{\alpha}(x) \right]

    with :math:`C_0^{\alpha}(x) = 1` and :math:`C_1^{\alpha}(x) = 2\alpha x`.

    **Numerical Stability:**
    The recurrence is stable for :math:`|x| \leq 1` and moderate degrees.
    For very high degrees or :math:`|x|` close to 1, consider normalization.

    **Normalization:**
    This implementation uses the standard (unnormalized) Gegenbauer polynomials.
    For orthonormal basis, multiply by :math:`\sqrt{\frac{(2l+2\alpha)\Gamma(l+2\alpha)}{l!\Gamma(2\alpha)}}`.

    Example
    -------
    >>> import tensorflow as tf
    >>> from arnold.layers.core.geometric import HypersphericalHarmonics
    >>>
    >>> # 3D spherical harmonics (n=3)
    >>> layer = HypersphericalHarmonics(dimension=3, max_degree=8, units=32)
    >>> x = tf.random.uniform((16, 10), -1, 1)
    >>> y = layer(x)  # shape: (16, 32)
    >>>
    >>> # 4D hyperspherical (n=4, quaternion space)
    >>> layer_4d = HypersphericalHarmonics(dimension=4, max_degree=6, units=64)
    """

    def __init__(
        self,
        *,
        units: int,
        dimension: int = 3,
        max_degree: int = 8,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        **kwargs,
    ):
        if dimension < 2:
            raise ValueError(f"dimension must be >= 2, got {dimension}")

        self.dimension = dimension
        # Gegenbauer parameter: α = (n-2)/2
        self.alpha = (dimension - 2) / 2.0

        super().__init__(
            units=units,
            max_degree=max_degree,
            input_clip=input_clip,
            **kwargs,
        )

    def _get_num_basis_functions(self) -> int:
        """Return number of basis functions: max_degree + 1 Gegenbauer polynomials."""
        return self.max_degree + 1

    def geometric_basis(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Compute Gegenbauer polynomial basis (zonal hyperspherical harmonics).

        Uses the three-term recurrence relation:

        .. math::

            l C_l^{\alpha}(x) = 2(l + \alpha - 1) x C_{l-1}^{\alpha}(x)
                - (l + 2\alpha - 2) C_{l-2}^{\alpha}(x)

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
        alpha = tf.cast(self.alpha, tf.float64)

        # Handle special case α = 0 (Chebyshev T_n limit)
        # For α → 0: lim C_l^α(x) / (2α) = T_l(x) / l for l > 0
        # We use a small epsilon to avoid division by zero
        use_chebyshev_limit = self.alpha < 1e-10

        # C_0^α(x) = 1
        C0 = tf.ones_like(x, dtype=tf.float64)
        basis = [C0]

        if self.max_degree >= 1:
            if use_chebyshev_limit:
                # Chebyshev T_1(x) = x (limit of C_1^0(x) * something)
                C1 = x
            else:
                # C_1^α(x) = 2αx
                C1 = 2.0 * alpha * x
            basis.append(C1)

            C_prev2 = C0  # C_{l-2}
            C_prev1 = C1  # C_{l-1}

            for l in range(2, self.max_degree + 1):
                l_f = tf.cast(l, tf.float64)

                if use_chebyshev_limit:
                    # Chebyshev recurrence: T_l(x) = 2x T_{l-1}(x) - T_{l-2}(x)
                    C_l = 2.0 * x * C_prev1 - C_prev2
                else:
                    # Gegenbauer recurrence:
                    # l C_l^α(x) = 2(l + α - 1) x C_{l-1}^α(x) - (l + 2α - 2) C_{l-2}^α(x)
                    coeff1 = 2.0 * (l_f + alpha - 1.0)
                    coeff2 = l_f + 2.0 * alpha - 2.0
                    C_l = (coeff1 * x * C_prev1 - coeff2 * C_prev2) / l_f

                basis.append(C_l)
                C_prev2 = C_prev1
                C_prev1 = C_l

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"dimension": self.dimension})
        return config

    @property
    def gegenbauer_alpha(self) -> float:
        """Return the Gegenbauer parameter α = (n-2)/2."""
        return self.alpha
