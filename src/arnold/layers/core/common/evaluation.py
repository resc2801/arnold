# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Evaluation strategies for polynomial basis functions.

This module provides algorithms for evaluating orthogonal polynomials:

- :func:`three_term_recurrence` — Standard forward recurrence
- :func:`clenshaw_eval` — Backward-stable Clenshaw algorithm
- :func:`horner_eval` — Horner's method for monomial basis

Mathematical Background
-----------------------
Most orthogonal polynomials satisfy a three-term recurrence:

.. math::
    P_{n+1}(x) = (a_n x + b_n) P_n(x) - c_n P_{n-1}(x)

The Clenshaw algorithm evaluates a sum of basis functions:

.. math::
    S(x) = \sum_{k=0}^{n} c_k P_k(x)

using backward recurrence, which is numerically more stable for
Chebyshev-like bases.

References
----------
.. [1] Clenshaw, C. W. (1955). "A note on the summation of Chebyshev series".
       Mathematical Tables and Other Aids to Computation, 9(51), 118-120.
.. [2] Press, W. H., et al. (2007). "Numerical Recipes". Cambridge University Press.
"""

from __future__ import annotations

from typing import Callable, Tuple, Optional

import tensorflow as tf


def three_term_recurrence(
    x: tf.Tensor,
    degree: int,
    alpha: Callable[[int], tf.Tensor],
    beta: Callable[[int], tf.Tensor],
    gamma: Callable[[int], tf.Tensor],
    p0: Optional[tf.Tensor] = None,
    p1_func: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
) -> tf.Tensor:
    r"""Evaluate polynomials via three-term recurrence.
    
    Computes :math:`P_0(x), P_1(x), \ldots, P_n(x)` where:
    
    .. math::
        P_{k+1}(x) = (\alpha_k x + \beta_k) P_k(x) - \gamma_k P_{k-1}(x)
    
    Parameters
    ----------
    x : tf.Tensor
        Input tensor of shape ``(...,)``.
    degree : int
        Maximum degree to compute.
    alpha : Callable[[int], tf.Tensor]
        Function returning :math:`\alpha_k` for degree k.
    beta : Callable[[int], tf.Tensor]
        Function returning :math:`\beta_k` for degree k.
    gamma : Callable[[int], tf.Tensor]
        Function returning :math:`\gamma_k` for degree k.
    p0 : tf.Tensor, optional
        Initial value :math:`P_0(x)`. Defaults to ones.
    p1_func : Callable, optional
        Function to compute :math:`P_1(x)` from x. Defaults to :math:`x`.
    
    Returns
    -------
    tf.Tensor
        Stacked polynomials of shape ``(..., degree + 1)``.
    
    Notes
    -----
    This function uses ``tf.while_loop`` for XLA compatibility.
    The recurrence is computed in forward direction, which may
    accumulate round-off errors for high degrees.
    
    For Chebyshev-like bases, prefer :func:`clenshaw_eval` when
    evaluating linear combinations of basis functions.
    
    Examples
    --------
    Legendre polynomials:
    
    >>> def alpha(n): return tf.constant((2*n + 1) / (n + 1), dtype=tf.float32)
    >>> def beta(n): return tf.constant(0.0, dtype=tf.float32)
    >>> def gamma(n): return tf.constant(n / (n + 1), dtype=tf.float32)
    >>> x = tf.constant([0.0, 0.5, 1.0])
    >>> P = three_term_recurrence(x, degree=3, alpha=alpha, beta=beta, gamma=gamma)
    """
    dtype = x.dtype
    
    # Initial conditions
    if p0 is None:
        p0 = tf.ones_like(x)
    
    if degree == 0:
        return tf.expand_dims(p0, axis=-1)
    
    # P_1(x)
    if p1_func is not None:
        p1 = p1_func(x)
    else:
        # Default: P_1(x) = (alpha_0 * x + beta_0) * P_0 - gamma_0 * P_{-1}
        # With P_{-1} = 0, this simplifies
        p1 = (alpha(0) * x + beta(0)) * p0
    
    if degree == 1:
        return tf.stack([p0, p1], axis=-1)
    
    # Build full sequence using tf.while_loop for XLA compatibility
    def body(k, p_prev, p_curr, result):
        a_k = tf.cast(alpha(k), dtype)
        b_k = tf.cast(beta(k), dtype)
        c_k = tf.cast(gamma(k), dtype)
        
        p_next = (a_k * x + b_k) * p_curr - c_k * p_prev
        
        # Update result tensor
        indices = tf.reshape(k + 1, [1])
        result = tf.tensor_scatter_nd_update(
            result,
            tf.expand_dims(indices, 1),
            tf.expand_dims(p_next, 0)
        )
        
        return k + 1, p_curr, p_next, result
    
    def cond(k, p_prev, p_curr, result):
        return k < degree
    
    # Initialize result tensor
    shape = tf.concat([tf.shape(x), [degree + 1]], axis=0)
    result = tf.zeros(shape, dtype=dtype)
    
    # Set P_0 and P_1
    result = tf.tensor_scatter_nd_update(
        result, [[0]], tf.expand_dims(p0, 0)
    )
    result = tf.tensor_scatter_nd_update(
        result, [[1]], tf.expand_dims(p1, 0)
    )
    
    # Run recurrence
    _, _, _, result = tf.while_loop(
        cond, body,
        loop_vars=[1, p0, p1, result],
        shape_invariants=[
            tf.TensorShape([]),
            x.shape,
            x.shape,
            tf.TensorShape(None),
        ]
    )
    
    # Transpose to get (..., degree + 1) shape
    perm = list(range(len(result.shape)))
    perm = perm[1:] + [0]
    return tf.transpose(result, perm)


def clenshaw_eval(
    x: tf.Tensor,
    coefficients: tf.Tensor,
    alpha: Callable[[int], tf.Tensor],
    beta: Callable[[int], tf.Tensor],
    gamma: Callable[[int], tf.Tensor],
) -> tf.Tensor:
    r"""Clenshaw algorithm for evaluating polynomial sums.
    
    Computes:
    
    .. math::
        S(x) = \sum_{k=0}^{n} c_k P_k(x)
    
    using backward recurrence, which is numerically stable for
    Chebyshev-like polynomial families.
    
    Parameters
    ----------
    x : tf.Tensor
        Evaluation points of shape ``(...,)``.
    coefficients : tf.Tensor
        Expansion coefficients :math:`c_k` of shape ``(n+1,)`` or ``(..., n+1)``.
    alpha, beta, gamma : Callable[[int], tf.Tensor]
        Recurrence coefficients as in :func:`three_term_recurrence`.
    
    Returns
    -------
    tf.Tensor
        Sum :math:`S(x)` of shape ``(...,)``.
    
    Notes
    -----
    The algorithm initializes :math:`b_{n+1} = b_{n+2} = 0` and computes:
    
    .. math::
        b_k = c_k + (\alpha_k x + \beta_k) b_{k+1} - \gamma_{k+1} b_{k+2}
    
    The final result is :math:`S(x) = b_0 P_0(x)` (with P_0 = 1 typically).
    
    This is particularly effective for Chebyshev expansions where the
    recurrence is well-conditioned.
    
    References
    ----------
    .. [1] Clenshaw, C. W. (1955). MTAC 9, 118-120.
    """
    dtype = x.dtype
    n = tf.shape(coefficients)[-1] - 1
    
    # Initialize b_{n+1} = b_{n+2} = 0
    b_next = tf.zeros_like(x)
    b_curr = tf.zeros_like(x)
    
    # Backward recurrence
    def body(k, b_curr, b_next):
        c_k = tf.gather(coefficients, k, axis=-1)
        a_k = tf.cast(alpha(k), dtype)
        beta_k = tf.cast(beta(k), dtype)
        gamma_k1 = tf.cast(gamma(k + 1), dtype)
        
        b_prev = c_k + (a_k * x + beta_k) * b_curr - gamma_k1 * b_next
        return k - 1, b_prev, b_curr
    
    def cond(k, b_curr, b_next):
        return k >= 0
    
    _, b_0, _ = tf.while_loop(
        cond, body,
        loop_vars=[n, b_curr, b_next]
    )
    
    return b_0


def horner_eval(x: tf.Tensor, coefficients: tf.Tensor) -> tf.Tensor:
    r"""Horner's method for polynomial evaluation.
    
    Evaluates a polynomial in monomial form:
    
    .. math::
        P(x) = c_0 + c_1 x + c_2 x^2 + \cdots + c_n x^n
    
    using Horner's rule:
    
    .. math::
        P(x) = c_0 + x(c_1 + x(c_2 + \cdots + x \cdot c_n))
    
    Parameters
    ----------
    x : tf.Tensor
        Evaluation points of shape ``(...,)``.
    coefficients : tf.Tensor
        Polynomial coefficients :math:`[c_0, c_1, \ldots, c_n]`.
    
    Returns
    -------
    tf.Tensor
        Polynomial values of shape ``(...,)``.
    
    Notes
    -----
    Horner's method requires exactly :math:`n` multiplications and
    :math:`n` additions, which is optimal for monomial evaluation.
    
    For orthogonal polynomial bases, prefer the three-term recurrence
    or Clenshaw algorithm instead.
    """
    n = tf.shape(coefficients)[-1]
    
    # Start from highest degree
    result = tf.gather(coefficients, n - 1, axis=-1) * tf.ones_like(x)
    
    def body(k, result):
        c_k = tf.gather(coefficients, k, axis=-1)
        result = result * x + c_k
        return k - 1, result
    
    def cond(k, result):
        return k >= 0
    
    _, result = tf.while_loop(
        cond, body,
        loop_vars=[n - 2, result]
    )
    
    return result


__all__ = [
    "three_term_recurrence",
    "clenshaw_eval",
    "horner_eval",
]
