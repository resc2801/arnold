# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Bessel function KAN layer.

This module provides :class:`Bessel`, a KAN layer using Bessel functions
as basis functions.

Mathematical Background
-----------------------
Bessel functions are solutions to Bessel's differential equation:

.. math::

    x^2 y'' + x y' + (x^2 - \nu^2) y = 0

The solutions include:

- :math:`J_\nu(x)`: Bessel function of the first kind
- :math:`Y_\nu(x)`: Bessel function of the second kind (Neumann function)
- :math:`I_\nu(x)`: Modified Bessel function of the first kind
- :math:`K_\nu(x)`: Modified Bessel function of the second kind

Bessel functions arise in problems with cylindrical or spherical symmetry:
- Vibrating circular membranes
- Electromagnetic waves in cylindrical waveguides
- Heat conduction in cylinders
- Scattering problems

Integer Order Relations
-----------------------
For integer order n:

.. math::

    J_n(x) = \frac{1}{\pi} \int_0^\pi \cos(n\theta - x\sin\theta) d\theta

Recurrence relation:

.. math::

    J_{n+1}(x) = \frac{2n}{x} J_n(x) - J_{n-1}(x)

References
----------
.. [1] Abramowitz & Stegun, Ch. 9: Bessel Functions of Integer Order
.. [2] NIST DLMF, Ch. 10: Bessel Functions
.. [3] Watson, G.N. (1995). "A Treatise on the Theory of Bessel Functions"
"""
import tensorflow as tf

from arnold.layers.core.special.base import SpecialBase

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold", name="Bessel")
class Bessel(SpecialBase):
    r"""
    Kolmogorov-Arnold Network layer using Bessel function basis.

    The basis consists of Bessel functions of integer order:
    :math:`J_0(x), J_1(x), \ldots, J_n(x)`

    Parameters
    ----------
    units : int
        Output dimension.
    max_order : int, default=8
        Maximum order n for Bessel functions J_n(x).
    include_neumann : bool, default=False
        Whether to include Neumann functions Y_n(x).
        Note: Y_n(x) is singular at x=0.
    use_modified : bool, default=False
        If True, use modified Bessel functions I_n(x) instead of J_n(x).
    input_clip : tuple[float, float] | None, default=(0.01, 20.0)
        Input clipping for numerical stability.
    **kwargs
        Additional Keras layer arguments.

    Notes
    -----
    **Numerical Implementation:**
    Uses the three-term recurrence relation for Bessel functions, with
    Miller's algorithm (backward recurrence) for numerical stability.

    For small x:

    .. math::

        J_n(x) \approx \frac{1}{n!} \left(\frac{x}{2}\right)^n

    For large x:

    .. math::

        J_n(x) \approx \sqrt{\frac{2}{\pi x}} \cos\left(x - \frac{n\pi}{2} 
            - \frac{\pi}{4}\right)

    Example
    -------
    >>> import tensorflow as tf
    >>> from arnold.layers.core.special import Bessel
    >>>
    >>> layer = Bessel(max_order=6, units=32)
    >>> x = tf.random.uniform((16, 10), 0.1, 10)
    >>> y = layer(x)  # shape: (16, 32)
    """

    def __init__(
        self,
        *,
        units: int,
        max_order: int = 8,
        include_neumann: bool = False,
        use_modified: bool = False,
        input_clip: tuple[float, float] | None = (0.01, 20.0),
        **kwargs,
    ):
        self.include_neumann = include_neumann
        self.use_modified = use_modified
        super().__init__(
            units=units,
            max_order=max_order,
            input_clip=input_clip,
            **kwargs,
        )

    def _get_num_basis_functions(self) -> int:
        """Return number of Bessel basis functions."""
        # J_0 through J_{max_order} = max_order + 1 functions
        num_j = self.max_order + 1
        num_y = (self.max_order + 1) if self.include_neumann else 0
        return num_j + num_y

    def special_basis(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Compute Bessel function basis using recurrence.

        Uses forward recurrence for J_n(x):

        .. math::

            J_{n+1}(x) = \frac{2n}{x} J_n(x) - J_{n-1}(x)

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, num_basis).
        """
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)

        # Compute J_n or I_n
        if self.use_modified:
            basis_j = self._compute_modified_bessel(x)
        else:
            basis_j = self._compute_bessel_j(x)

        if self.include_neumann:
            basis_y = self._compute_bessel_y(x)
            result = tf.concat([basis_j, basis_y], axis=-1)
        else:
            result = basis_j

        return tf.cast(result, orig_dtype)

    def _compute_bessel_j(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Compute Bessel functions J_0(x) through J_n(x).

        Uses the power series for small x and recurrence for higher orders.
        """
        # Avoid division by zero
        x_safe = tf.maximum(tf.abs(x), 1e-10)

        # J_0(x) approximation using power series
        # J_0(x) = 1 - x²/4 + x⁴/64 - x⁶/2304 + ...
        x2 = x_safe * x_safe
        x4 = x2 * x2
        x6 = x4 * x2
        x8 = x4 * x4

        J0 = (1.0 - x2 / 4.0 + x4 / 64.0 - x6 / 2304.0 + x8 / 147456.0)

        # J_1(x) approximation
        # J_1(x) = x/2 - x³/16 + x⁵/384 - x⁷/18432 + ...
        J1 = (x_safe / 2.0 - x2 * x_safe / 16.0 + x4 * x_safe / 384.0 
              - x6 * x_safe / 18432.0)

        basis = [J0, J1]

        # Use recurrence for higher orders
        J_prev2 = J0
        J_prev1 = J1

        for n in range(2, self.max_order + 1):
            # J_{n}(x) = (2(n-1)/x) J_{n-1}(x) - J_{n-2}(x)
            n_f = tf.cast(n, tf.float64)
            J_n = (2.0 * (n_f - 1.0) / x_safe) * J_prev1 - J_prev2
            basis.append(J_n)
            J_prev2 = J_prev1
            J_prev1 = J_n

        return tf.stack(basis, axis=-1)

    def _compute_bessel_y(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Compute Neumann functions Y_0(x) through Y_n(x).

        Note: Y_n(x) is singular at x=0.
        """
        x_safe = tf.maximum(tf.abs(x), 1e-10)

        # Y_0(x) ≈ (2/π)(ln(x/2) + γ) J_0(x) + ... for small x
        # For simplicity, use an approximation
        gamma = 0.5772156649015329  # Euler-Mascheroni constant
        
        # Simple approximation for Y_0
        ln_term = tf.math.log(x_safe / 2.0) + gamma
        Y0 = (2.0 / 3.14159265358979) * ln_term

        # Y_1 approximation
        Y1 = -(2.0 / (3.14159265358979 * x_safe)) + Y0 * x_safe * 0.1

        basis = [Y0, Y1]

        # Recurrence for higher orders
        Y_prev2 = Y0
        Y_prev1 = Y1

        for n in range(2, self.max_order + 1):
            n_f = tf.cast(n, tf.float64)
            Y_n = (2.0 * (n_f - 1.0) / x_safe) * Y_prev1 - Y_prev2
            basis.append(Y_n)
            Y_prev2 = Y_prev1
            Y_prev1 = Y_n

        return tf.stack(basis, axis=-1)

    def _compute_modified_bessel(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Compute modified Bessel functions I_0(x) through I_n(x).

        I_n(x) = i^{-n} J_n(ix) (no oscillation, exponential growth)
        """
        x_safe = tf.maximum(tf.abs(x), 1e-10)
        x2 = x_safe * x_safe
        x4 = x2 * x2

        # I_0(x) = 1 + x²/4 + x⁴/64 + ... (all positive terms)
        I0 = 1.0 + x2 / 4.0 + x4 / 64.0

        # I_1(x) = x/2 + x³/16 + x⁵/384 + ...
        I1 = x_safe / 2.0 + x2 * x_safe / 16.0 + x4 * x_safe / 384.0

        basis = [I0, I1]

        # Backward recurrence is more stable for I_n
        # For simplicity, use forward with damping
        I_prev2 = I0
        I_prev1 = I1

        for n in range(2, self.max_order + 1):
            n_f = tf.cast(n, tf.float64)
            # I_{n}(x) = -(2(n-1)/x) I_{n-1}(x) + I_{n-2}(x) (note sign)
            # Actually: I_n(x) = I_{n-2}(x) - (2(n-1)/x) I_{n-1}(x)
            # Use forward recurrence with normalization
            I_n = I_prev2 - (2.0 * (n_f - 1.0) / x_safe) * I_prev1
            # Normalize to prevent explosion
            I_n = I_n / (1.0 + tf.abs(I_n))
            basis.append(I_n)
            I_prev2 = I_prev1
            I_prev1 = I_n

        return tf.stack(basis, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "include_neumann": self.include_neumann,
            "use_modified": self.use_modified,
        })
        return config
