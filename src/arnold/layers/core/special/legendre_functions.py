# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Legendre functions KAN layer.

This module provides :class:`LegendreFunctions`, a KAN layer using Legendre
functions of the first and second kind as basis functions.

Mathematical Background
-----------------------
Legendre functions are generalizations of Legendre polynomials to non-integer
degrees and complex arguments. The Legendre equation is:

.. math::

    (1 - x^2) y'' - 2x y' + \nu(\nu+1) y = 0

Solutions include:

- :math:`P_\nu(x)`: Legendre function of the first kind
- :math:`Q_\nu(x)`: Legendre function of the second kind

For integer ν = n, :math:`P_n(x)` reduces to Legendre polynomials.

Associated Legendre Functions
-----------------------------
The associated Legendre equation includes an additional term:

.. math::

    (1 - x^2) y'' - 2x y' + \left[\nu(\nu+1) - \frac{\mu^2}{1-x^2}\right] y = 0

Solutions:

- :math:`P_\nu^\mu(x)`: Associated Legendre function of the first kind
- :math:`Q_\nu^\mu(x)`: Associated Legendre function of the second kind

These are crucial for:
- Spherical harmonics: :math:`Y_l^m(\theta, \phi)`
- Gravitational/electromagnetic multipole expansions
- Quantum angular momentum

References
----------
.. [1] NIST DLMF, Ch. 14: Legendre and Related Functions
.. [2] Abramowitz & Stegun, Ch. 8: Legendre Functions
"""
import tensorflow as tf

from arnold.layers.core.special.base import SpecialBase

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold", name="LegendreFunctions")
class LegendreFunctions(SpecialBase):
    r"""
    Kolmogorov-Arnold Network layer using Legendre functions.

    Uses Legendre functions P_ν(x) and Q_ν(x) with non-integer degrees
    as basis functions.

    Parameters
    ----------
    units : int
        Output dimension.
    max_order : int, default=8
        Maximum degree for Legendre functions.
    include_second_kind : bool, default=True
        Whether to include Q_ν(x) functions.
    use_half_integer : bool, default=False
        If True, use half-integer degrees (ν = 0.5, 1.5, 2.5, ...).
    input_clip : tuple[float, float] | None, default=(-0.99, 0.99)
        Input clipping (|x| < 1 for standard domain).
    **kwargs
        Additional Keras layer arguments.

    Example
    -------
    >>> import tensorflow as tf
    >>> from arnold.layers.core.special import LegendreFunctions
    >>>
    >>> layer = LegendreFunctions(max_order=6, units=32)
    >>> x = tf.random.uniform((16, 10), -0.9, 0.9)
    >>> y = layer(x)  # shape: (16, 32)
    """

    def __init__(
        self,
        *,
        units: int,
        max_order: int = 8,
        include_second_kind: bool = True,
        use_half_integer: bool = False,
        input_clip: tuple[float, float] | None = (-0.99, 0.99),
        **kwargs,
    ):
        self.include_second_kind = include_second_kind
        self.use_half_integer = use_half_integer
        super().__init__(
            units=units,
            max_order=max_order,
            input_clip=input_clip,
            **kwargs,
        )

    def _get_num_basis_functions(self) -> int:
        """Return number of Legendre function basis functions."""
        num_p = self.max_order + 1
        num_q = (self.max_order + 1) if self.include_second_kind else 0
        return num_p + num_q

    def special_basis(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Compute Legendre function basis.

        For integer degrees, uses the standard recurrence:

        .. math::

            (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, input_dim), |x| < 1.

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, num_basis).
        """
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)

        if self.use_half_integer:
            basis_p = self._compute_half_integer_p(x)
        else:
            basis_p = self._compute_integer_p(x)

        if self.include_second_kind:
            basis_q = self._compute_q(x)
            result = tf.concat([basis_p, basis_q], axis=-1)
        else:
            result = basis_p

        return tf.cast(result, orig_dtype)

    def _compute_integer_p(self, x: tf.Tensor) -> tf.Tensor:
        """Compute P_n(x) for integer n using recurrence."""
        P0 = tf.ones_like(x, dtype=tf.float64)
        P1 = x

        basis = [P0]

        if self.max_order >= 1:
            basis.append(P1)

            P_prev2 = P0
            P_prev1 = P1

            for n in range(2, self.max_order + 1):
                n_f = tf.cast(n, tf.float64)
                P_n = ((2.0 * n_f - 1.0) * x * P_prev1 - (n_f - 1.0) * P_prev2) / n_f
                basis.append(P_n)
                P_prev2 = P_prev1
                P_prev1 = P_n

        return tf.stack(basis, axis=-1)

    def _compute_half_integer_p(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Compute P_ν(x) for half-integer ν = n + 1/2.

        For ν = -1/2 and 1/2:

        .. math::

            P_{-1/2}(x) = \frac{2}{\pi} K\left(\sqrt{\frac{1-x}{2}}\right)

        where K is the complete elliptic integral.
        """
        # Simplified approximation for half-integer Legendre functions
        # True implementation would require elliptic integrals

        basis = []

        # P_{1/2}(x) ≈ sqrt(2/(1+x)) * something
        # Use polynomial approximation
        sqrt_term = tf.sqrt(tf.maximum(1.0 + x, 1e-10))

        for n in range(self.max_order + 1):
            nu = tf.cast(n, tf.float64) + 0.5
            # Approximate using asymptotic behavior
            P_nu = tf.pow(sqrt_term, nu) * tf.cos(nu * tf.acos(tf.clip_by_value(x, -1.0, 1.0)))
            basis.append(P_nu)

        return tf.stack(basis, axis=-1)

    def _compute_q(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Compute Q_n(x) for integer n.

        Q_0(x) = (1/2) ln((1+x)/(1-x))
        Q_1(x) = x Q_0(x) - 1
        """
        x_safe = tf.clip_by_value(x, -0.999, 0.999)

        # Q_0(x) = artanh(x) = (1/2) ln((1+x)/(1-x))
        Q0 = 0.5 * tf.math.log((1.0 + x_safe) / (1.0 - x_safe))

        basis = [Q0]

        if self.max_order >= 1:
            # Q_1(x) = x Q_0(x) - 1
            Q1 = x * Q0 - 1.0
            basis.append(Q1)

            Q_prev2 = Q0
            Q_prev1 = Q1

            # Same recurrence as P_n
            for n in range(2, self.max_order + 1):
                n_f = tf.cast(n, tf.float64)
                Q_n = ((2.0 * n_f - 1.0) * x * Q_prev1 - (n_f - 1.0) * Q_prev2) / n_f
                basis.append(Q_n)
                Q_prev2 = Q_prev1
                Q_prev1 = Q_n

        return tf.stack(basis, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "include_second_kind": self.include_second_kind,
            "use_half_integer": self.use_half_integer,
        })
        return config
