# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Numerical utilities for stable computation.

This module provides numerically robust primitives for common operations
that are prone to overflow, underflow, or catastrophic cancellation.

Stability Considerations
------------------------
Many polynomial evaluations involve operations that can fail numerically:

- **Factorials/Gamma functions**: Overflow quickly for moderate n
- **Logarithms**: Undefined for negative or zero inputs
- **Division**: Catastrophic for near-zero denominators
- **Summation**: Accumulation of round-off errors

We provide safe wrappers and alternative formulations:

- :func:`safe_log` — Log with domain extension
- :func:`safe_sqrt` — Square root with epsilon floor
- :func:`safe_divide` — Division with denominator clamping
- :func:`log_pochhammer` — Rising factorial in log domain
- :func:`kahan_sum` — Compensated summation
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf


def safe_log(
    x: tf.Tensor,
    epsilon: float = 1e-37,
) -> tf.Tensor:
    r"""Logarithm with numerical safeguard.
    
    Computes :math:`\log(\max(x, \epsilon))` to avoid :math:`-\infty`.
    
    Parameters
    ----------
    x : tf.Tensor
        Input tensor.
    epsilon : float
        Minimum value floor. Default ``1e-37`` (near float32 min).
    
    Returns
    -------
    tf.Tensor
        Logarithm of x, floored at log(epsilon).
    
    Notes
    -----
    For float64, consider using ``epsilon=1e-300``.
    """
    eps = tf.constant(epsilon, dtype=x.dtype)
    return tf.math.log(tf.maximum(x, eps))


def safe_sqrt(
    x: tf.Tensor,
    epsilon: float = 1e-12,
) -> tf.Tensor:
    r"""Square root with numerical safeguard.
    
    Computes :math:`\sqrt{\max(x, \epsilon)}`.
    
    Parameters
    ----------
    x : tf.Tensor
        Input tensor.
    epsilon : float
        Minimum value floor.
    
    Returns
    -------
    tf.Tensor
        Square root of x, safeguarded against negative values.
    """
    eps = tf.constant(epsilon, dtype=x.dtype)
    return tf.sqrt(tf.maximum(x, eps))


def safe_divide(
    numerator: tf.Tensor,
    denominator: tf.Tensor,
    epsilon: float = 1e-12,
) -> tf.Tensor:
    r"""Division with denominator clamping.
    
    Computes :math:`\frac{a}{b}` where :math:`|b|` is clamped away from zero.
    
    Parameters
    ----------
    numerator : tf.Tensor
        Numerator tensor.
    denominator : tf.Tensor
        Denominator tensor.
    epsilon : float
        Minimum absolute value for denominator.
    
    Returns
    -------
    tf.Tensor
        Safe quotient.
    
    Notes
    -----
    The sign of the denominator is preserved:
    
    .. math::
        \text{safe\_divide}(a, b) = \frac{a}{\text{sign}(b) \cdot \max(|b|, \epsilon)}
    """
    eps = tf.constant(epsilon, dtype=denominator.dtype)
    sign = tf.sign(denominator)
    # Handle zero sign case
    sign = tf.where(tf.equal(sign, 0), tf.ones_like(sign), sign)
    safe_denom = sign * tf.maximum(tf.abs(denominator), eps)
    return numerator / safe_denom


def log_pochhammer(
    a: tf.Tensor,
    n: int,
) -> tf.Tensor:
    r"""Rising factorial (Pochhammer symbol) in log domain.
    
    Computes:
    
    .. math::
        \log (a)_n = \log \Gamma(a + n) - \log \Gamma(a)
    
    where :math:`(a)_n = a(a+1)(a+2)\cdots(a+n-1)`.
    
    Parameters
    ----------
    a : tf.Tensor
        Base value.
    n : int
        Number of terms.
    
    Returns
    -------
    tf.Tensor
        Log of rising factorial.
    
    Notes
    -----
    This avoids overflow for large n by staying in log domain.
    The identity :math:`(a)_n = \Gamma(a+n)/\Gamma(a)` is used.
    
    For negative integer a, the result may be ``-inf`` or ``nan``.
    """
    if n == 0:
        return tf.zeros_like(a)
    
    a_plus_n = a + tf.cast(n, a.dtype)
    return tf.math.lgamma(a_plus_n) - tf.math.lgamma(a)


def log_factorial(n: tf.Tensor) -> tf.Tensor:
    r"""Logarithm of factorial using log-gamma.
    
    Computes :math:`\log(n!) = \log \Gamma(n + 1)`.
    
    Parameters
    ----------
    n : tf.Tensor
        Non-negative integer values.
    
    Returns
    -------
    tf.Tensor
        Log of factorial.
    """
    return tf.math.lgamma(tf.cast(n, tf.float64) + 1.0)


def log_binomial(n: tf.Tensor, k: tf.Tensor) -> tf.Tensor:
    r"""Logarithm of binomial coefficient.
    
    Computes:
    
    .. math::
        \log \binom{n}{k} = \log \Gamma(n+1) - \log \Gamma(k+1) - \log \Gamma(n-k+1)
    
    Parameters
    ----------
    n : tf.Tensor
        Total count.
    k : tf.Tensor
        Selection count.
    
    Returns
    -------
    tf.Tensor
        Log of binomial coefficient.
    
    Notes
    -----
    This is stable for large n, k where direct computation would overflow.
    """
    n_float = tf.cast(n, tf.float64)
    k_float = tf.cast(k, tf.float64)
    return (
        tf.math.lgamma(n_float + 1.0)
        - tf.math.lgamma(k_float + 1.0)
        - tf.math.lgamma(n_float - k_float + 1.0)
    )


def kahan_sum(values: tf.Tensor, axis: int = -1) -> tf.Tensor:
    r"""Kahan compensated summation.
    
    Reduces accumulated round-off error when summing many floating-point
    numbers by tracking a compensation term.
    
    Parameters
    ----------
    values : tf.Tensor
        Values to sum.
    axis : int
        Axis along which to sum.
    
    Returns
    -------
    tf.Tensor
        Compensated sum.
    
    Notes
    -----
    The algorithm maintains:
    
    .. math::
        c = (sum + y) - sum - y
    
    where c captures the lost low-order bits. This is particularly
    important for high-degree polynomial sums.
    
    Warning
    -------
    XLA compilation may reorder operations, potentially defeating
    the compensation mechanism. Use with caution in jit-compiled code.
    """
    # Move sum axis to last position for easier iteration
    ndims = len(values.shape)
    if axis < 0:
        axis = ndims + axis
    
    perm = list(range(ndims))
    perm.remove(axis)
    perm.append(axis)
    values_t = tf.transpose(values, perm)
    
    n = tf.shape(values_t)[-1]
    
    def body(i, total, compensation):
        y = values_t[..., i] - compensation
        t = total + y
        compensation = (t - total) - y
        return i + 1, t, compensation
    
    def cond(i, total, compensation):
        return i < n
    
    _, result, _ = tf.while_loop(
        cond, body,
        loop_vars=[
            0,
            tf.zeros(tf.shape(values_t)[:-1], dtype=values.dtype),
            tf.zeros(tf.shape(values_t)[:-1], dtype=values.dtype),
        ]
    )
    
    return result


def stabilize_recurrence(
    p_prev: tf.Tensor,
    p_curr: tf.Tensor,
    threshold: float = 1e30,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    r"""Rescale recurrence values to prevent overflow.
    
    When polynomial values grow beyond threshold, rescale both
    P_{n-1} and P_n by the same factor.
    
    Parameters
    ----------
    p_prev : tf.Tensor
        Previous polynomial value :math:`P_{n-1}(x)`.
    p_curr : tf.Tensor
        Current polynomial value :math:`P_n(x)`.
    threshold : float
        Maximum allowed magnitude before rescaling.
    
    Returns
    -------
    tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        Rescaled (p_prev, p_curr, scale_factor).
    
    Notes
    -----
    This is important for polynomials like Hermite which grow as
    :math:`O(2^n n!)` and can easily overflow even for moderate degrees.
    """
    max_val = tf.maximum(tf.reduce_max(tf.abs(p_prev)), tf.reduce_max(tf.abs(p_curr)))
    scale = tf.where(
        max_val > threshold,
        max_val / threshold,
        tf.ones_like(max_val),
    )
    return p_prev / scale, p_curr / scale, scale


__all__ = [
    "safe_log",
    "safe_sqrt",
    "safe_divide",
    "log_pochhammer",
    "log_factorial",
    "log_binomial",
    "kahan_sum",
    "stabilize_recurrence",
]
