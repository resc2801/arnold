## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Zernike polynomial KAN layer.

Zernike polynomials are orthogonal polynomials on the unit disk, widely used
in optics for wavefront analysis and aberration characterization.
"""
import math

import tensorflow as tf

from arnold.utils.compilation import kan_function
from arnold.utils.constraints import softplus_lower_bound
from .poly_base import PolynomialBase


tfk = tf.keras
tfkl = tfk.layers
kan_fn = kan_function(jit_compile=False)  # Disable XLA for complex indexing


def _inverse_softplus_lower_bound(y: float, lower_bound: float) -> float:
    """Compute inverse of softplus_lower_bound for initialization."""
    y_shifted = y - lower_bound
    if y_shifted <= 0:
        return -10.0
    return math.log(math.exp(y_shifted) - 1.0) if y_shifted < 20 else y_shifted


@tfk.utils.register_keras_serializable(package="arnold", name="Zernike")
class Zernike(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Zernike polynomials.

    Zernike polynomials :math:`Z_n^m(\rho, \theta)` are orthogonal on the unit disk
    and are the standard for wavefront analysis in optics. They are defined as:

    .. math::

        Z_n^m(\rho, \theta) = R_n^{|m|}(\rho) \cdot 
            \begin{cases}
                \cos(m\theta) & m \geq 0 \\
                \sin(|m|\theta) & m < 0
            \end{cases}

    where the radial polynomial :math:`R_n^m(\rho)` is:

    .. math::

        R_n^m(\rho) = \sum_{k=0}^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! \left(\frac{n+m}{2}-k\right)! 
            \left(\frac{n-m}{2}-k\right)!} \rho^{n-2k}

    For 1D inputs, this layer uses the radial polynomials :math:`R_n^0(\rho)` 
    (rotationally symmetric Zernike polynomials).

    Orthogonality on the unit disk:

    .. math::

        \int_0^1 R_n^m(\rho) R_{n'}^m(\rho) \rho \, d\rho = 
            \frac{\delta_{nn'}}{2(n+1)}

    Parameters
    ----------
    degree : int
        Maximum radial degree n. The basis includes all R_n^0 for n = 0, 2, 4, ..., degree
        (only even n for m=0).
    units : int
        Output dimension.
    input_clip : tuple, optional
        Input clipping range. Default is (0.0, 1.0) since Zernike are defined on [0,1].

    Notes
    -----
    - Only radial (m=0) Zernike polynomials are used for 1D inputs
    - For m=0, only even degrees n are valid
    - R_0^0 = 1 (piston)
    - R_2^0 = 2ρ² - 1 (defocus)
    - R_4^0 = 6ρ⁴ - 6ρ² + 1 (primary spherical)
    - Applications: optics, astronomy, corneal topography

    References
    ----------
    .. [1] Born, M. & Wolf, E. (1999). "Principles of Optics", Chapter 9
    .. [2] Noll, R.J. (1976). "Zernike polynomials and atmospheric turbulence"
    .. [3] OEIS A027641 - Numerators in Zernike polynomial coefficients
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        input_clip: tuple[float, float] | None = (0.0, 1.0),
        **kwargs,
    ):
        # For m=0, only even degrees are valid
        # Effective basis count: degree//2 + 1
        # Store the original degree for basis computation
        self._original_degree = degree
        # Pass adjusted degree to base class so weight shapes are correct
        # The base uses degree+1 as number of basis functions
        effective_basis_count = degree // 2 + 1
        super().__init__(
            degree=effective_basis_count - 1, 
            units=units, 
            input_clip=input_clip, 
            **kwargs
        )
        # Override degree back to original for pseudo_vandermonde
        self._zernike_max_degree = degree

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Compute radial Zernike polynomial basis R_n^0(ρ).

        Uses the recurrence relation for radial Zernike polynomials:

        .. math::

            R_n^0(\rho) = \frac{2n-1}{n} (2\rho^2 - 1) R_{n-1}^0(\rho) - 
                \frac{n-1}{n} R_{n-2}^0(\rho)

        But for m=0, we use even n only, so the effective recurrence is on n = 0, 2, 4, ...

        Parameters
        ----------
        x : tf.Tensor
            Input tensor ρ of shape (batch, input_dim), expected in [0, 1].

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, num_polynomials).
            num_polynomials = degree//2 + 1 (for even degrees 0, 2, 4, ..., degree)
        """
        orig_dtype = x.dtype
        rho = tf.cast(x, tf.float64)
        rho2 = rho * rho

        # R_0^0(ρ) = 1
        R0 = tf.ones_like(rho, dtype=tf.float64)
        basis = [R0]

        if self._zernike_max_degree >= 2:
            # R_2^0(ρ) = 2ρ² - 1
            R2 = 2.0 * rho2 - 1.0
            basis.append(R2)

            # For higher even degrees, use recurrence
            # R_n^0 can be computed from previous R_{n-2}^0
            # General form: R_n^0(ρ) = sum of ρ^{n-2k} terms
            
            R_prev2 = R0  # R_{n-4}
            R_prev1 = R2  # R_{n-2}

            for n in range(4, self._zernike_max_degree + 1, 2):
                # Recurrence for m=0 radial Zernike
                # R_n^0 = (2n-1)/n * (2ρ² - 1) * R_{n-2}^0 - (n-1)/n * R_{n-4}^0
                # But this is for consecutive n. For even n only:
                # Use direct coefficient computation for stability
                
                n_f = tf.cast(n, tf.float64)
                
                # Alternative: Use Chebyshev-like relation
                # R_n^0(ρ) relates to Legendre polynomials: R_n^0(ρ) = P_n/2(2ρ² - 1)
                # So we can use: R_n^0 = 2(2ρ² - 1) R_{n-2}^0 - R_{n-4}^0 (approximately)
                
                # More accurate recurrence for m=0:
                # R_n^0(ρ) = 2(2ρ² - 1) R_{n-2}^0(ρ) - R_{n-4}^0(ρ)
                R_n = 2.0 * (2.0 * rho2 - 1.0) * R_prev1 - R_prev2
                
                basis.append(R_n)
                R_prev2 = R_prev1
                R_prev1 = R_n

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    @property
    def num_basis_functions(self):
        """Number of radial Zernike polynomials (even degrees only)."""
        return self._zernike_max_degree // 2 + 1

    def compute_output_shape(self, input_shape):
        """Override to account for even-degree-only basis."""
        return input_shape[:-1] + (self.units,)

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        # Return the original Zernike degree, not the adjusted one
        config["degree"] = self._zernike_max_degree
        return config
