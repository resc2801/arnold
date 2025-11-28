# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Whittaker function KAN layer.

This module provides :class:`Whittaker`, a KAN layer using Whittaker functions
as basis functions.

Mathematical Background
-----------------------
Whittaker functions are solutions to Whittaker's differential equation:

.. math::

    \frac{d^2 W}{dx^2} + \left(-\frac{1}{4} + \frac{\kappa}{x} 
        + \frac{1/4 - \mu^2}{x^2}\right) W = 0

The two independent solutions are:

- :math:`M_{\kappa,\mu}(x)`: Regular at x=0
- :math:`W_{\kappa,\mu}(x)`: Behaves well as x→∞

These functions are related to:
- Confluent hypergeometric functions
- Coulomb wave functions (hydrogen atom)
- Kummer functions

References
----------
.. [1] NIST DLMF, Ch. 13: Confluent Hypergeometric Functions
.. [2] Whittaker, E.T. & Watson, G.N. "A Course of Modern Analysis"
"""
import tensorflow as tf

from arnold.layers.core.special.base import SpecialBase

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold", name="Whittaker")
class Whittaker(SpecialBase):
    r"""
    Kolmogorov-Arnold Network layer using Whittaker functions.

    Uses Whittaker M and W functions as basis.

    Parameters
    ----------
    units : int
        Output dimension.
    max_order : int, default=8
        Maximum order for parameter variations.
    mu : float, default=0.5
        Whittaker parameter μ.
    input_clip : tuple[float, float] | None, default=(0.1, 20.0)
        Input clipping for numerical stability.
    **kwargs
        Additional Keras layer arguments.

    Example
    -------
    >>> import tensorflow as tf
    >>> from arnold.layers.core.special import Whittaker
    >>>
    >>> layer = Whittaker(max_order=4, mu=0.5, units=32)
    >>> x = tf.random.uniform((16, 10), 0.5, 10)
    >>> y = layer(x)
    """

    def __init__(
        self,
        *,
        units: int,
        max_order: int = 8,
        mu: float = 0.5,
        input_clip: tuple[float, float] | None = (0.1, 20.0),
        **kwargs,
    ):
        self.mu = mu
        super().__init__(
            units=units,
            max_order=max_order,
            input_clip=input_clip,
            **kwargs,
        )

    def _get_num_basis_functions(self) -> int:
        """Return number of Whittaker basis functions."""
        # M and W for different κ values
        return 2 * (self.max_order + 1)

    def special_basis(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Compute Whittaker function basis.

        Uses the relation to confluent hypergeometric functions:

        .. math::

            M_{\kappa,\mu}(x) = e^{-x/2} x^{\mu+1/2} M(1/2+\mu-\kappa, 1+2\mu, x)

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, input_dim), x > 0.

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, num_basis).
        """
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        mu = tf.cast(self.mu, tf.float64)

        x_safe = tf.maximum(x, 1e-6)

        # Common factor
        exp_factor = tf.exp(-x_safe / 2.0)
        power_factor = tf.pow(x_safe, mu + 0.5)

        basis = []

        for k in range(self.max_order + 1):
            kappa = tf.cast(k, tf.float64) * 0.5  # κ = 0, 0.5, 1, 1.5, ...

            # Simplified M_{κ,μ}(x) approximation
            # M_{κ,μ}(x) ≈ exp(-x/2) x^{μ+1/2} (1 + O(x))
            a = 0.5 + mu - kappa
            b = 1.0 + 2.0 * mu

            # First-order confluent hypergeometric approximation
            # M(a, b, x) ≈ 1 + (a/b)x + (a(a+1))/(b(b+1)) x²/2 + ...
            M_approx = 1.0 + (a / b) * x_safe
            if b * (b + 1.0) > 1e-10:
                M_approx = M_approx + (a * (a + 1.0)) / (b * (b + 1.0)) * x_safe * x_safe / 2.0

            M_kappa_mu = exp_factor * power_factor * M_approx
            basis.append(M_kappa_mu)

            # W_{κ,μ}(x) - asymptotically e^{-x/2} x^κ
            W_kappa_mu = exp_factor * tf.pow(x_safe, kappa) * (1.0 + 0.1 / x_safe)
            basis.append(W_kappa_mu)

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "mu": self.mu,
        })
        return config
