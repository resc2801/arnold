# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Continuous q-Legendre polynomials.

This module implements Continuous q-Legendre polynomials :math:`P_n(x | q)`.

Mathematical Background
-----------------------
The Continuous q-Legendre polynomials are the special case of continuous
q-ultraspherical polynomials with :math:`\beta = q^{1/2}`:

.. math::

    P_n(x | q) = C_n(x; q^{1/2} | q)

They satisfy a three-term recurrence:

.. math::

    2x(1 - q^{n+1/2}) P_n(x) = (1 - q^{n+1}) P_{n+1}(x) + (1 - q^n) P_{n-1}(x)

with :math:`P_0(x | q) = 1` and :math:`P_1(x | q) = 2x(1-q^{1/2})/(1-q)`.

Properties
----------
- Special case β = q^{1/2} of continuous q-ultraspherical
- Limit: :math:`\lim_{q \to 1} P_n(x | q) = P_n(x)` (Legendre)
- Symmetric: :math:`P_n(-x | q) = (-1)^n P_n(x | q)`
- Simpler coefficients than general q-ultraspherical

References
----------
.. [1] NIST DLMF 18.28(v)
.. [2] Koekoek et al. (2010), Chapter 14.10.1
"""

import tensorflow as tf

from arnold.utils.compilation import kan_function

from .base import QPolynomialBase


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="ContinuousQLegendre")
class ContinuousQLegendre(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Continuous q-Legendre polynomials.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.
    units : int
        Output dimensionality.
    q : float
        Base parameter in (0, 1). Default 0.5.
    q_trainable : bool
        Whether q is trainable.

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
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        **kwargs,
    ):
        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Compute Continuous q-Legendre polynomial basis.

        Uses the three-term recurrence with β = q^{1/2}:

        .. math::

            2x(1 - q^{n+1/2}) P_n = (1 - q^{n+1}) P_{n+1} + (1 - q^n) P_{n-1}

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
        sqrt_q = tf.sqrt(q)

        # P_0(x) = 1
        P0 = tf.ones_like(x, dtype=tf.float64)
        basis = [P0]

        if self.degree >= 1:
            # P_1(x) = 2x(1 - sqrt(q)) / (1 - q)
            P1 = 2.0 * x * (1.0 - sqrt_q) / (1.0 - q + 1e-12)
            basis.append(P1)

            # Recurrence: 2x(1 - q^{n+1/2}) P_n = (1 - q^{n+1}) P_{n+1} + (1 - q^n) P_{n-1}
            # Rearranged: P_{n+1} = [2x(1 - q^{n+1/2}) P_n - (1 - q^n) P_{n-1}] / (1 - q^{n+1})
            P_prev = P0
            P_curr = P1

            for n in range(1, self.degree):
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)
                q_np1 = q_n * q
                q_n_half = q_n * sqrt_q  # q^{n+1/2}

                # Coefficients
                A_coeff = 2.0 * (1.0 - q_n_half)  # Multiplies x * P_n
                C_coeff = 1.0 - q_n  # Multiplies P_{n-1}
                denom = 1.0 - q_np1

                P_next = (A_coeff * x * P_curr - C_coeff * P_prev) / (denom + 1e-12)
                basis.append(P_next)
                P_prev = P_curr
                P_curr = P_next

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)


__all__ = [
    "ContinuousQLegendre",
]
