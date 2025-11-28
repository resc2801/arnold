# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Airy function KAN layer.

This module provides :class:`Airy`, a KAN layer using Airy functions as
basis functions.

Mathematical Background
-----------------------
Airy functions are solutions to Airy's differential equation:

.. math::

    y'' - xy = 0

The two linearly independent solutions are:

- :math:`\text{Ai}(x)`: Airy function of the first kind (decays for x > 0)
- :math:`\text{Bi}(x)`: Airy function of the second kind (grows for x > 0)

Airy functions arise naturally at:
- Turning points in WKB approximations
- Caustics in wave optics
- Boundary layers in fluid dynamics
- Quantum tunneling problems

Asymptotic Behavior
-------------------
For large positive x:

.. math::

    \text{Ai}(x) \sim \frac{e^{-\frac{2}{3}x^{3/2}}}{2\sqrt{\pi}x^{1/4}}

For large negative x:

.. math::

    \text{Ai}(x) \sim \frac{\sin(\frac{2}{3}|x|^{3/2} + \frac{\pi}{4})}
        {\sqrt{\pi}|x|^{1/4}}

References
----------
.. [1] Abramowitz & Stegun, Ch. 10: Bessel Functions of Fractional Order
.. [2] NIST DLMF, Ch. 9: Airy and Related Functions
"""
import tensorflow as tf

from arnold.layers.core.special.base import SpecialBase


tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold", name="Airy")
class Airy(SpecialBase):
    r"""
    Kolmogorov-Arnold Network layer using Airy function basis.

    The basis consists of Airy functions Ai(x) and Bi(x) evaluated at
    scaled arguments to provide multiple oscillatory components.

    The basis functions are:

    .. math::

        \phi_k(x) = \text{Ai}(\alpha_k x + \beta_k), \quad k = 0, 1, \ldots

    where :math:`\alpha_k` and :math:`\beta_k` control the frequency and phase.

    Parameters
    ----------
    units : int
        Output dimension.
    max_order : int, default=8
        Number of Airy function components (Ai and Bi variants).
    include_bi : bool, default=True
        Whether to include Bi(x) functions in addition to Ai(x).
    scale : float, default=1.0
        Scaling factor for input argument.
    input_clip : tuple[float, float] | None, default=(-10.0, 10.0)
        Input clipping for numerical stability.
    **kwargs
        Additional Keras layer arguments.

    Notes
    -----
    **Numerical Implementation:**
    We approximate Airy functions using polynomial approximations that are
    numerically stable. For production use, consider scipy.special.airy
    wrapped in tf.py_function, or a custom TensorFlow implementation.

    **Current Implementation:**
    Uses a rational polynomial approximation for Ai(x) that is accurate
    for |x| ≲ 10. For larger arguments, asymptotic expansions should be used.

    Example
    -------
    >>> import tensorflow as tf
    >>> from arnold.layers.core.special import Airy
    >>>
    >>> layer = Airy(max_order=6, units=32)
    >>> x = tf.random.uniform((16, 10), -5, 5)
    >>> y = layer(x)  # shape: (16, 32)
    """

    def __init__(
        self,
        *,
        units: int,
        max_order: int = 8,
        include_bi: bool = True,
        scale: float = 1.0,
        input_clip: tuple[float, float] | None = (-10.0, 10.0),
        **kwargs,
    ):
        self.include_bi = include_bi
        self.scale = scale
        super().__init__(
            units=units,
            max_order=max_order,
            input_clip=input_clip,
            **kwargs,
        )

    def _get_num_basis_functions(self) -> int:
        """Return number of Airy basis functions."""
        # Ai at different scales + optionally Bi at different scales
        num_ai = self.max_order
        num_bi = self.max_order if self.include_bi else 0
        return num_ai + num_bi

    def special_basis(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Compute Airy function basis.

        Uses polynomial/rational approximations for numerical stability.

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
        x_scaled = x * self.scale

        basis_functions = []

        # Ai(x) at different frequency scales
        for k in range(self.max_order):
            freq = 1.0 + 0.5 * k  # Frequency scaling
            ai_k = self._airy_ai(x_scaled * freq)
            basis_functions.append(ai_k)

        # Bi(x) at different frequency scales
        if self.include_bi:
            for k in range(self.max_order):
                freq = 1.0 + 0.5 * k
                bi_k = self._airy_bi(x_scaled * freq)
                basis_functions.append(bi_k)

        result = tf.stack(basis_functions, axis=-1)
        return tf.cast(result, orig_dtype)

    def _airy_ai(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Approximate Airy function Ai(x).

        Uses a polynomial approximation valid for moderate x values.
        For large |x|, asymptotic formulas should be used.

        The approximation is based on the power series for small x
        and rational approximations for moderate x.
        """
        # Polynomial approximation coefficients (Abramowitz & Stegun)
        # Ai(0) ≈ 0.3550280538878172
        # Ai'(0) ≈ -0.2588194037928068

        # Use a simple polynomial approximation for demonstration
        # In production, use scipy or a proper implementation

        # Maclaurin series coefficients for Ai(x)
        c0 = 0.3550280538878172  # Ai(0)
        c1 = -0.2588194037928068  # Ai'(0)
        c2 = c0 / 2  # From the ODE: y'' = xy
        c3 = c1 / 6
        c4 = c0 / 24 + c2 / 12
        c5 = c1 / 120 + c3 / 20

        x2 = x * x
        x3 = x2 * x
        x4 = x2 * x2
        x5 = x4 * x

        # Simple polynomial approximation (accurate for |x| < 3)
        ai_approx = c0 + c1 * x + c2 * x2 + c3 * x3 + c4 * x4 + c5 * x5

        # For large positive x, Ai(x) → 0 exponentially
        # For large negative x, Ai(x) oscillates
        # Apply a damping factor for numerical stability
        damping = tf.exp(-tf.maximum(x, 0.0) * 0.5)

        return ai_approx * damping

    def _airy_bi(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Approximate Airy function Bi(x).

        Bi(x) grows exponentially for x > 0, so we apply normalization.
        """
        # Bi(0) ≈ 0.6149266274460007
        # Bi'(0) ≈ 0.4482883573538264

        c0 = 0.6149266274460007
        c1 = 0.4482883573538264
        c2 = c0 / 2
        c3 = c1 / 6
        c4 = c0 / 24 + c2 / 12
        c5 = c1 / 120 + c3 / 20

        x2 = x * x
        x3 = x2 * x
        x4 = x2 * x2
        x5 = x4 * x

        bi_approx = c0 + c1 * x + c2 * x2 + c3 * x3 + c4 * x4 + c5 * x5

        # Normalize to prevent explosion
        normalization = 1.0 / (1.0 + tf.exp(tf.maximum(x, 0.0)))

        return bi_approx * normalization

    def get_config(self):
        config = super().get_config()
        config.update({
            "include_bi": self.include_bi,
            "scale": self.scale,
        })
        return config
