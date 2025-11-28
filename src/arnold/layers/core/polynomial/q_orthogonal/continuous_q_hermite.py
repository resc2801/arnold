# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Continuous q-Hermite polynomials.

This module implements Continuous q-Hermite polynomials :math:`H_n(x | q)`.

Mathematical Background
-----------------------
The Continuous q-Hermite polynomials are defined as:

.. math::

    H_n(\cos\theta | q) = \sum_{\ell=0}^{n} \frac{(q; q)_n}{(q; q)_\ell (q; q)_{n-\ell}} 
        e^{i(n-2\ell)\theta}

They satisfy the three-term recurrence:

.. math::

    2x \, H_n(x | q) = H_{n+1}(x | q) + (1 - q^n) \, H_{n-1}(x | q)

with :math:`H_0(x | q) = 1` and :math:`H_1(x | q) = 2x`.

Properties
----------
- Also known as Rogers-Szegő polynomials
- Orthogonal on :math:`[-1, 1]` with q-theta weight
- Limit: :math:`\lim_{q \to 1} H_n(x\sqrt{(1-q)/2} | q) / ((1-q)/2)^{n/2} = H_n(x)`

References
----------
.. [1] NIST DLMF 18.28(vi)
.. [2] Koekoek et al. (2010), Chapter 14.26
"""

import tensorflow as tf

from arnold.utils.compilation import kan_function

from .base import QPolynomialBase


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="ContinuousQHermite")
class ContinuousQHermite(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Continuous q-Hermite polynomials.

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
    .. [1] NIST DLMF 18.28(vi)
    .. [2] Koekoek et al. (2010), Chapter 14.26
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
        Compute Continuous q-Hermite polynomial basis.

        Uses the three-term recurrence:

        .. math::

            H_{n+1}(x) = 2x \cdot H_n(x) - (1 - q^n) \cdot H_{n-1}(x)

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

        # H_0(x) = 1
        H0 = tf.ones_like(x, dtype=tf.float64)
        basis = [H0]

        if self.degree >= 1:
            # H_1(x) = 2x
            H1 = 2.0 * x
            basis.append(H1)

            # Recurrence: H_{n+1}(x) = 2x * H_n(x) - (1 - q^n) * H_{n-1}(x)
            H_prev = H0
            H_curr = H1

            for n in range(1, self.degree):
                q_n = tf.pow(q, tf.cast(n, tf.float64))
                C_n = 1.0 - q_n  # Coefficient of H_{n-1}

                H_next = 2.0 * x * H_curr - C_n * H_prev
                basis.append(H_next)
                H_prev = H_curr
                H_curr = H_next

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)


__all__ = [
    "ContinuousQHermite",
]
