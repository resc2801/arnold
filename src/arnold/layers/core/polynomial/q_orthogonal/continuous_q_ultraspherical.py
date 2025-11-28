# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Continuous q-Ultraspherical (Rogers) polynomials.

This module implements Continuous q-Ultraspherical polynomials :math:`C_n(x; \beta | q)`.

Mathematical Background
-----------------------
The Continuous q-Ultraspherical polynomials are defined as:

.. math::

    C_n(\cos\theta; \beta | q) = \sum_{\ell=0}^{n} 
        \frac{(\beta; q)_\ell (\beta; q)_{n-\ell}}{(q; q)_\ell (q; q)_{n-\ell}} 
        e^{i(n-2\ell)\theta}

They satisfy the three-term recurrence:

.. math::

    2x(1 - \beta q^n) C_n(x) = (1 - q^{n+1}) C_{n+1}(x) + (1 - \beta^2 q^{n-1}) C_{n-1}(x)

with :math:`C_0(x; \beta | q) = 1` and :math:`C_1(x; \beta | q) = 2x(1-\beta)/(1-q)`.

Properties
----------
- Also known as Rogers polynomials
- Limit: :math:`\lim_{q \to 1} C_n(x; q^\lambda | q) = C_n^{(\lambda)}(x)` (Gegenbauer)
- Special case β = 0: :math:`C_n(x; 0 | q) = H_n(x | q) / (q; q)_n` (q-Hermite)
- Symmetric: :math:`C_n(-x; \beta | q) = (-1)^n C_n(x; \beta | q)`

References
----------
.. [1] NIST DLMF 18.28(v)
.. [2] Koekoek et al. (2010), Chapter 14.10.1
"""

import math

import tensorflow as tf

from arnold.utils.compilation import kan_function

from .base import QPolynomialBase


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="ContinuousQUltraspherical")
class ContinuousQUltraspherical(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Continuous q-Ultraspherical polynomials.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.
    units : int
        Output dimensionality.
    beta : float
        Parameter β with |β| < 1. Default 0.5.
    beta_trainable : bool
        Whether β is trainable.
    q : float
        Base parameter in (0, 1). Default 0.5.

    References
    ----------
    .. [1] NIST DLMF 18.28(v)
    .. [2] Koekoek et al. (2010), Chapter 14.10.1
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        beta: float = 0.5,
        beta_trainable: bool = False,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        **kwargs,
    ):
        if not -1 < beta < 1:
            raise ValueError(f"beta must be in (-1, 1), got {beta}")

        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )

        self.beta_init = beta
        self.beta_trainable = beta_trainable
        self.beta_logits = None

    def build(self, input_shape):
        super().build(input_shape)

        # β ∈ (-1, 1) via scaled tanh: β = tanh(logit)
        # Inverse: logit = arctanh(β)
        beta_logit = 0.5 * math.log((1 + self.beta_init) / (1 - self.beta_init + 1e-12))

        self.beta_logits = self.add_weight(
            name="beta_logits",
            shape=(),
            initializer=tf.constant_initializer(beta_logit),
            trainable=self.beta_trainable,
            dtype=self.dtype,
        )

    def _get_beta(self, dtype=tf.float64):
        """Get β parameter constrained to (-1, 1) via tanh."""
        return tf.tanh(tf.cast(self.beta_logits, dtype))

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Compute Continuous q-Ultraspherical polynomial basis.

        Uses the three-term recurrence:

        .. math::

            2x(1 - \beta q^n) C_n = (1 - q^{n+1}) C_{n+1} + (1 - \beta^2 q^{n-1}) C_{n-1}

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
        beta = self._get_beta(tf.float64)

        # C_0(x) = 1
        C0 = tf.ones_like(x, dtype=tf.float64)
        basis = [C0]

        if self.degree >= 1:
            # C_1(x) = 2x(1 - β) / (1 - q)
            # For numerical stability, use direct formula
            C1 = 2.0 * x * (1.0 - beta) / (1.0 - q + 1e-12)
            basis.append(C1)

            # Recurrence: 2x(1 - β*q^n) C_n = (1 - q^{n+1}) C_{n+1} + (1 - β²*q^{n-1}) C_{n-1}
            # Rearranged: C_{n+1} = [2x(1 - β*q^n) C_n - (1 - β²*q^{n-1}) C_{n-1}] / (1 - q^{n+1})
            C_prev = C0
            C_curr = C1

            for n in range(1, self.degree):
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)
                q_np1 = q_n * q
                q_nm1 = q_n / (q + 1e-12)

                # Coefficients
                A_coeff = 2.0 * (1.0 - beta * q_n)  # Multiplies x * C_n
                C_coeff = 1.0 - beta * beta * q_nm1  # Multiplies C_{n-1}
                denom = 1.0 - q_np1

                C_next = (A_coeff * x * C_curr - C_coeff * C_prev) / (denom + 1e-12)
                basis.append(C_next)
                C_prev = C_curr
                C_curr = C_next

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "beta": self.beta_init,
            "beta_trainable": self.beta_trainable,
        })
        return config


__all__ = [
    "ContinuousQUltraspherical",
]
