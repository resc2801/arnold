# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Zernike polynomial KAN layer.

Zernike polynomials are orthogonal polynomials on the unit disk, widely used
in optics for wavefront analysis and aberration characterization.

Mathematical Background
-----------------------
Zernike polynomials :math:`Z_n^m(\rho, \theta)` are defined on the unit disk as:

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

Applications:
- Optics & wavefront analysis
- Corneal topography
- Aberration characterization
- Astronomy (atmospheric turbulence)

References
----------
.. [1] Born, M. & Wolf, E. (1999). "Principles of Optics", Chapter 9
.. [2] Noll, R.J. (1976). "Zernike polynomials and atmospheric turbulence"
.. [3] OEIS A027641 - Numerators in Zernike polynomial coefficients
"""
import tensorflow as tf

from arnold.layers.core.geometric.base import GeometricBase

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold", name="Zernike")
class Zernike(GeometricBase):
    r"""
    Kolmogorov-Arnold Network layer using Zernike polynomials.

    Zernike polynomials are orthogonal on the unit disk with the weight
    function :math:`w(\rho) = \rho`. This layer uses the radial polynomials
    :math:`R_n^0(\rho)` for 1D inputs.

    Orthogonality:

    .. math::

        \int_0^1 R_n^0(\rho) R_{n'}^0(\rho) \rho \, d\rho = 
            \frac{\delta_{nn'}}{2(n+1)}

    Parameters
    ----------
    degree : int
        Maximum radial degree n. The basis includes :math:`R_n^0` for 
        n = 0, 2, 4, ..., degree (only even n for m=0).
    units : int
        Output dimension.
    input_clip : tuple[float, float] | None, default=(0.0, 1.0)
        Input clipping range. Zernike polynomials are defined on [0, 1].
    use_bias : bool, default=True
        Whether to add a bias term.
    activation : str or callable, default=None
        Activation function.
    kernel_regularizer : regularizer, default=None
        Regularizer for polynomial coefficients.

    Notes
    -----
    - Only radial (m=0) Zernike polynomials are used for 1D inputs
    - For m=0, only even degrees n are valid
    - :math:`R_0^0(\rho) = 1` (piston)
    - :math:`R_2^0(\rho) = 2\rho^2 - 1` (defocus)
    - :math:`R_4^0(\rho) = 6\rho^4 - 6\rho^2 + 1` (primary spherical)

    Example
    -------
    >>> import tensorflow as tf
    >>> from arnold.layers.core.geometric import Zernike
    >>>
    >>> layer = Zernike(degree=8, units=32)
    >>> x = tf.random.uniform((16, 10), 0, 1)  # ρ in [0, 1]
    >>> y = layer(x)  # shape: (16, 32)
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        input_clip: tuple[float, float] | None = (0.0, 1.0),
        **kwargs,
    ):
        if degree < 0:
            raise ValueError(f"degree must be >= 0, got {degree}")

        self.degree = degree
        # For m=0, only even degrees are valid: 0, 2, 4, ..., degree
        # Number of basis functions: degree//2 + 1
        super().__init__(
            units=units,
            max_degree=degree,
            input_clip=input_clip,
            **kwargs,
        )

    def _get_num_basis_functions(self) -> int:
        """Return number of radial Zernike polynomials (even degrees only)."""
        return self.degree // 2 + 1

    def geometric_basis(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Compute radial Zernike polynomial basis :math:`R_n^0(\rho)`.

        Uses the direct formula for radial Zernike polynomials with m=0:

        .. math::

            R_n^0(\rho) = \sum_{k=0}^{n/2} \frac{(-1)^k (n-k)!}
                {k! \left(\frac{n}{2}-k\right)! \left(\frac{n}{2}-k\right)!} \rho^{n-2k}

        For numerical stability, coefficients are precomputed and the
        polynomial is evaluated using Horner's method.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor ρ of shape (batch, input_dim), expected in [0, 1].

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, num_polynomials).
        """
        orig_dtype = x.dtype
        rho = tf.cast(x, tf.float64)
        rho2 = rho * rho

        # R_0^0(ρ) = 1
        R0 = tf.ones_like(rho, dtype=tf.float64)
        basis = [R0]

        if self.degree >= 2:
            # R_2^0(ρ) = 2ρ² - 1
            R2 = 2.0 * rho2 - 1.0
            basis.append(R2)

        if self.degree >= 4:
            # R_4^0(ρ) = 6ρ⁴ - 6ρ² + 1
            rho4 = rho2 * rho2
            R4 = 6.0 * rho4 - 6.0 * rho2 + 1.0
            basis.append(R4)

        if self.degree >= 6:
            # R_6^0(ρ) = 20ρ⁶ - 30ρ⁴ + 12ρ² - 1
            rho4 = rho2 * rho2
            rho6 = rho4 * rho2
            R6 = 20.0 * rho6 - 30.0 * rho4 + 12.0 * rho2 - 1.0
            basis.append(R6)

        if self.degree >= 8:
            # R_8^0(ρ) = 70ρ⁸ - 140ρ⁶ + 90ρ⁴ - 20ρ² + 1
            rho4 = rho2 * rho2
            rho6 = rho4 * rho2
            rho8 = rho4 * rho4
            R8 = 70.0 * rho8 - 140.0 * rho6 + 90.0 * rho4 - 20.0 * rho2 + 1.0
            basis.append(R8)

        if self.degree >= 10:
            # R_10^0(ρ) = 252ρ¹⁰ - 630ρ⁸ + 560ρ⁶ - 210ρ⁴ + 30ρ² - 1
            rho4 = rho2 * rho2
            rho6 = rho4 * rho2
            rho8 = rho4 * rho4
            rho10 = rho8 * rho2
            R10 = (252.0 * rho10 - 630.0 * rho8 + 560.0 * rho6 
                   - 210.0 * rho4 + 30.0 * rho2 - 1.0)
            basis.append(R10)

        if self.degree >= 12:
            # R_12^0(ρ) = 924ρ¹² - 2772ρ¹⁰ + 3150ρ⁸ - 1680ρ⁶ + 420ρ⁴ - 42ρ² + 1
            rho4 = rho2 * rho2
            rho6 = rho4 * rho2
            rho8 = rho4 * rho4
            rho10 = rho8 * rho2
            rho12 = rho6 * rho6
            R12 = (924.0 * rho12 - 2772.0 * rho10 + 3150.0 * rho8
                   - 1680.0 * rho6 + 420.0 * rho4 - 42.0 * rho2 + 1.0)
            basis.append(R12)

        # For higher degrees, use the general formula
        for n in range(14, self.degree + 1, 2):
            R_n = self._compute_R_n_0(rho, n)
            basis.append(R_n)

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def _compute_R_n_0(self, rho: tf.Tensor, n: int) -> tf.Tensor:
        r"""
        Compute R_n^0(ρ) using the direct sum formula.

        .. math::

            R_n^0(\rho) = \sum_{k=0}^{n/2} \frac{(-1)^k (n-k)!}
                {k! \left(\frac{n}{2}-k\right)!^2} \rho^{n-2k}

        Parameters
        ----------
        rho : tf.Tensor
            Input tensor ρ.
        n : int
            Degree (must be even).

        Returns
        -------
        tf.Tensor
            R_n^0(ρ).
        """
        import math

        half_n = n // 2
        result = tf.zeros_like(rho, dtype=tf.float64)

        for k in range(half_n + 1):
            # Coefficient: (-1)^k * (n-k)! / (k! * ((n/2)-k)!^2)
            sign = (-1) ** k
            num = math.factorial(n - k)
            denom = math.factorial(k) * (math.factorial(half_n - k) ** 2)
            coef = sign * num / denom

            # ρ^(n-2k)
            power = n - 2 * k
            result = result + coef * tf.pow(rho, power)

        return result

    def pseudo_vandermonde(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Compute the pseudo-Vandermonde matrix of Zernike polynomials.

        This is an alias for :meth:`geometric_basis`, provided for API
        compatibility with polynomial layers.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor ρ of shape (batch, input_dim), expected in [0, 1].

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, num_polynomials).
        """
        return self.geometric_basis(x)

    def get_config(self):
        config = super().get_config()
        # Remove max_degree (it's redundant with degree)
        config.pop("max_degree", None)
        config.update({"degree": self.degree})
        return config
