## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""
Constraint utilities for parameter transformations.

This module provides functions for mapping unconstrained parameter values to
constrained domains while maintaining smooth gradient flow. These are essential
for trainable parameters that must satisfy mathematical constraints (e.g.,
alpha > -0.5 for Gegenbauer polynomials).

The key insight is that storing parameters in an **unconstrained** space (logits)
and transforming them to the constrained space in the forward pass ensures:
1. Optimizers can update parameters freely without violating constraints
2. Gradients flow smoothly near constraint boundaries (unlike tf.maximum)
3. All valid constraint values are reachable from the parameter space
"""

from __future__ import annotations

import tensorflow as tf


def softplus_lower_bound(logits: tf.Tensor, lower_bound: float, eps: float = 1e-6) -> tf.Tensor:
    """
    Map unconstrained logits to (lower_bound + eps, ∞).

    Uses the transformation: result = softplus(logits) + lower_bound + eps

    This is suitable for parameters like:
    - Gegenbauer alpha (must be > -0.5)
    - Jacobi alpha/beta (must be > -1)
    - GeneralizedLaguerre alpha (must be > -1)

    Parameters
    ----------
    logits : tf.Tensor
        Unconstrained parameter values.
    lower_bound : float
        The exclusive lower bound for the output.
    eps : float
        Small positive offset to keep strictly above the bound.

    Returns
    -------
    tf.Tensor
        Values guaranteed to be > lower_bound + eps.

    Examples
    --------
    >>> # For Gegenbauer alpha > -0.5:
    >>> alpha = softplus_lower_bound(alpha_logits, lower_bound=-0.5)
    >>> # For Jacobi alpha > -1:
    >>> alpha = softplus_lower_bound(alpha_logits, lower_bound=-1.0)
    """
    return tf.nn.softplus(logits) + lower_bound + eps


def softplus_positive(logits: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
    """
    Map unconstrained logits to (eps, ∞).

    Equivalent to softplus_lower_bound(logits, 0.0, eps).

    Parameters
    ----------
    logits : tf.Tensor
        Unconstrained parameter values.
    eps : float
        Small positive offset to ensure strict positivity.

    Returns
    -------
    tf.Tensor
        Strictly positive values > eps.

    Examples
    --------
    >>> epsilon = softplus_positive(epsilon_logits)
    >>> scale = softplus_positive(scale_logits)
    """
    return tf.nn.softplus(logits) + eps


def sigmoid_interval(logits: tf.Tensor, low: float, high: float, eps: float = 1e-6) -> tf.Tensor:
    """
    Map unconstrained logits to (low + eps, high - eps).

    Uses the transformation: result = sigmoid(logits) * (high - low - 2*eps) + low + eps

    This is suitable for parameters that must lie in an open interval, such as:
    - AskeyWilson q (must satisfy |q| < 1)

    Parameters
    ----------
    logits : tf.Tensor
        Unconstrained parameter values.
    low : float
        The exclusive lower bound.
    high : float
        The exclusive upper bound.
    eps : float
        Small offset from boundaries.

    Returns
    -------
    tf.Tensor
        Values guaranteed to be in (low + eps, high - eps).

    Examples
    --------
    >>> # For q in (-1, 1):
    >>> q = sigmoid_interval(q_logits, low=-1.0, high=1.0)
    """
    range_width = high - low - 2 * eps
    return tf.nn.sigmoid(logits) * range_width + low + eps


def inverse_softplus(value: tf.Tensor) -> tf.Tensor:
    """
    Compute the inverse of softplus for initializing logits.

    Given a target positive value, returns the logits that would produce
    approximately that value when passed through softplus.

    softplus_inv(y) = log(exp(y) - 1) for y > 0

    Parameters
    ----------
    value : tf.Tensor
        Target positive values.

    Returns
    -------
    tf.Tensor
        Logits such that softplus(logits) ≈ value.
    """
    # For numerical stability, use log(exp(y) - 1) only when y is not too large
    # For large y, softplus(x) ≈ x, so inverse is just y
    return tf.where(
        value > 20.0,
        value,
        tf.math.log(tf.math.expm1(value))  # log(exp(y) - 1)
    )


def inverse_softplus_lower_bound(value: tf.Tensor, lower_bound: float, eps: float = 1e-6) -> tf.Tensor:
    """
    Compute logits that produce a target value after softplus_lower_bound.

    Parameters
    ----------
    value : tf.Tensor
        Target values (must be > lower_bound + eps).
    lower_bound : float
        The lower bound used in softplus_lower_bound.
    eps : float
        The epsilon used in softplus_lower_bound.

    Returns
    -------
    tf.Tensor
        Logits such that softplus_lower_bound(logits, lower_bound, eps) ≈ value.
    """
    shifted = value - lower_bound - eps
    return inverse_softplus(shifted)
