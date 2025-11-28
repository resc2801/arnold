# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Quantum q-Krawtchouk polynomials.

This module implements Quantum q-Krawtchouk polynomials :math:`K_n^{qtm}(x; p, N | q)`.

Mathematical Background
-----------------------
The Quantum q-Krawtchouk polynomials are a quantum-group-theoretic variant of
q-Krawtchouk polynomials arising in the representation theory of :math:`U_q(su(2))`.

They satisfy a three-term recurrence:

.. math::

    \lambda(x) K_n(x) = A_n K_{n+1}(x) + B_n K_n(x) + C_n K_{n-1}(x)

where the spectral variable :math:`\lambda(x)` and coefficients depend on
the quantum deformation parameter q.

Properties
----------
- Related to Clebsch-Gordan coefficients for :math:`U_q(su(2))`
- Limit q → 1: reduces to classical Krawtchouk
- Applications in quantum computing and quantum information theory

References
----------
.. [1] Koekoek et al. (2010), Chapter 14.15
.. [2] Koornwinder (1989). "Krawtchouk polynomials, a unification..."
"""

import tensorflow as tf

from arnold.utils.compilation import kan_function
from arnold.utils.constraints import softplus_lower_bound

from .base import QPolynomialBase, _inverse_softplus_lower_bound


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="QuantumQKrawtchouk")
class QuantumQKrawtchouk(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Quantum q-Krawtchouk polynomials.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.
    units : int
        Output dimensionality.
    p : float
        Parameter p > 0. Default 0.5.
    N : int
        Discrete support size, must be >= degree.
    p_trainable : bool
        Whether p is trainable.
    q : float
        Base parameter in (0, 1). Default 0.5.

    References
    ----------
    .. [1] Koekoek et al. (2010), Chapter 14.15
    .. [2] Koornwinder (1989)
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        p: float = 0.5,
        N: int = 10,
        p_trainable: bool = False,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        if N < degree:
            raise ValueError(f"N must be >= degree, got N={N}, degree={degree}")

        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )

        self.p_init = p
        self.N = N
        self.p_trainable = p_trainable
        self.p_logits = None

    def build(self, input_shape):
        super().build(input_shape)

        p_logit = _inverse_softplus_lower_bound(self.p_init, 0.0)

        self.p_logits = self.add_weight(
            name="p_logits",
            shape=(),
            initializer=tf.constant_initializer(p_logit),
            trainable=self.p_trainable,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Compute Quantum q-Krawtchouk polynomial basis using three-term recurrence.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, degree+1).
        """
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)

        q = self._get_q(tf.float64)
        p = softplus_lower_bound(tf.cast(self.p_logits, tf.float64), 0.0)
        N = tf.cast(self.N, tf.float64)

        ones = tf.ones_like(x, dtype=tf.float64)

        # K_0(x) = 1
        K0 = ones
        basis = [K0]

        if self.degree >= 1:
            # Three-term recurrence for Quantum q-Krawtchouk
            # Based on Koekoek et al. (2010)

            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for Quantum q-Krawtchouk."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)
                q_N = tf.pow(q, N)

                # A_n coefficient (of K_{n+1})
                A_n = (1.0 - q_n * q) * (1.0 - p * q_n * q)

                # C_n coefficient (of K_{n-1})
                C_n = p * q * (1.0 - q_n) * (q_N - q_n)

                # B_n = spectral normalization
                B_n = p * q + q_N - A_n - C_n

                return A_n, B_n, C_n

            # K_1(x)
            A0, B0, _ = compute_coeffs(0)
            K1 = (x - B0 * ones) / (A0 + 1e-12)
            basis.append(K1)

            # K_n for n >= 2
            K_prev = K0
            K_curr = K1

            for n in range(1, self.degree):
                A_n, B_n, C_n = compute_coeffs(n)
                K_next = (x * K_curr - B_n * K_curr - C_n * K_prev) / (A_n + 1e-12)
                basis.append(K_next)
                K_prev = K_curr
                K_curr = K_next

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "p": self.p_init,
            "N": self.N,
            "p_trainable": self.p_trainable,
        })
        return config


__all__ = [
    "QuantumQKrawtchouk",
]
