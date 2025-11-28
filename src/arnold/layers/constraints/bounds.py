# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Bounded constraint utilities and transformations.

This module provides functions for mapping unconstrained parameters to
bounded domains while maintaining smooth gradient flow.

Mathematical Background
-----------------------
The key transformations are:

1. **Softplus lower bound**: :math:`f(x) = \text{softplus}(x) + L + \epsilon`
   
   Maps :math:`\mathbb{R} \to (L + \epsilon, \infty)`

2. **Sigmoid interval**: :math:`f(x) = \sigma(x) \cdot (H - L - 2\epsilon) + L + \epsilon`
   
   Maps :math:`\mathbb{R} \to (L + \epsilon, H - \epsilon)`

These are preferred over hard constraints because:

- Gradients are well-defined everywhere (no zero-gradient regions)
- Optimizers can update parameters freely in unconstrained space
- All valid constraint values are reachable
"""

from __future__ import annotations

import tensorflow as tf

from arnold.layers.constraints.base import KANConstraint

tfk = tf.keras


# =============================================================================
# Transformation Functions
# =============================================================================


def softplus_lower_bound(
    logits: tf.Tensor, lower_bound: float, eps: float = 1e-6
) -> tf.Tensor:
    r"""
    Map unconstrained logits to :math:`(L + \epsilon, \infty)`.

    Uses the transformation:

    .. math::

        f(x) = \text{softplus}(x) + L + \epsilon

    This is suitable for parameters like:

    - Gegenbauer :math:`\alpha` (must be :math:`> -0.5`)
    - Jacobi :math:`\alpha, \beta` (must be :math:`> -1`)
    - GeneralizedLaguerre :math:`\alpha` (must be :math:`> -1`)

    Parameters
    ----------
    logits : tf.Tensor
        Unconstrained parameter values.
    lower_bound : float
        The exclusive lower bound :math:`L`.
    eps : float, default=1e-6
        Small positive offset to keep strictly above the bound.

    Returns
    -------
    tf.Tensor
        Values guaranteed to be :math:`> L + \epsilon`.

    Examples
    --------
    >>> # For Gegenbauer alpha > -0.5:
    >>> alpha = softplus_lower_bound(alpha_logits, lower_bound=-0.5)
    >>> # For Jacobi alpha > -1:
    >>> alpha = softplus_lower_bound(alpha_logits, lower_bound=-1.0)
    """
    return tf.nn.softplus(logits) + lower_bound + eps


def softplus_positive(logits: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
    r"""
    Map unconstrained logits to :math:`(\epsilon, \infty)`.

    Equivalent to ``softplus_lower_bound(logits, 0.0, eps)``.

    Parameters
    ----------
    logits : tf.Tensor
        Unconstrained parameter values.
    eps : float, default=1e-6
        Small positive offset to ensure strict positivity.

    Returns
    -------
    tf.Tensor
        Strictly positive values :math:`> \epsilon`.

    Examples
    --------
    >>> epsilon = softplus_positive(epsilon_logits)
    >>> scale = softplus_positive(scale_logits)
    """
    return tf.nn.softplus(logits) + eps


def sigmoid_interval(
    logits: tf.Tensor, low: float, high: float, eps: float = 1e-6
) -> tf.Tensor:
    r"""
    Map unconstrained logits to :math:`(L + \epsilon, H - \epsilon)`.

    Uses the transformation:

    .. math::

        f(x) = \sigma(x) \cdot (H - L - 2\epsilon) + L + \epsilon

    This is suitable for parameters that must lie in an open interval:

    - AskeyWilson :math:`q` (must satisfy :math:`|q| < 1`)
    - Parameters bounded by physical constraints

    Parameters
    ----------
    logits : tf.Tensor
        Unconstrained parameter values.
    low : float
        The exclusive lower bound :math:`L`.
    high : float
        The exclusive upper bound :math:`H`.
    eps : float, default=1e-6
        Small offset from boundaries.

    Returns
    -------
    tf.Tensor
        Values guaranteed to be in :math:`(L + \epsilon, H - \epsilon)`.

    Examples
    --------
    >>> # For q in (-1, 1):
    >>> q = sigmoid_interval(q_logits, low=-1.0, high=1.0)
    """
    range_width = high - low - 2 * eps
    return tf.nn.sigmoid(logits) * range_width + low + eps


# =============================================================================
# Inverse Transformations (for initialization)
# =============================================================================


def inverse_softplus(value: tf.Tensor) -> tf.Tensor:
    r"""
    Compute the inverse of softplus for initializing logits.

    Given a target positive value, returns the logits that would produce
    approximately that value when passed through softplus.

    .. math::

        \text{softplus}^{-1}(y) = \log(\exp(y) - 1) \quad \text{for } y > 0

    Parameters
    ----------
    value : tf.Tensor
        Target positive values.

    Returns
    -------
    tf.Tensor
        Logits such that :math:`\text{softplus}(\text{logits}) \approx \text{value}`.
    """
    # For numerical stability, use log(exp(y) - 1) only when y is not too large
    # For large y, softplus(x) ≈ x, so inverse is just y
    return tf.where(value > 20.0, value, tf.math.log(tf.math.expm1(value)))


def inverse_softplus_lower_bound(
    value: tf.Tensor, lower_bound: float, eps: float = 1e-6
) -> tf.Tensor:
    r"""
    Compute logits that produce a target value after softplus_lower_bound.

    Parameters
    ----------
    value : tf.Tensor
        Target values (must be :math:`> L + \epsilon`).
    lower_bound : float
        The lower bound :math:`L` used in softplus_lower_bound.
    eps : float, default=1e-6
        The epsilon used in softplus_lower_bound.

    Returns
    -------
    tf.Tensor
        Logits such that ``softplus_lower_bound(logits, L, eps) ≈ value``.
    """
    shifted = value - lower_bound - eps
    return inverse_softplus(shifted)


# =============================================================================
# Keras Constraint Class
# =============================================================================


@tfk.utils.register_keras_serializable(package="arnold")
class BoundedConstraint(KANConstraint):
    r"""
    Keras constraint for weights bounded to an interval.

    Projects weights into the interval :math:`[L, H]` using clipping.
    For differentiable constraints, use the transformation functions
    directly in the forward pass instead.

    Parameters
    ----------
    lower : float, default=-1.0
        Lower bound.
    upper : float, default=1.0
        Upper bound.

    Examples
    --------
    >>> constraint = BoundedConstraint(lower=0.0, upper=1.0)
    >>> layer = tf.keras.layers.Dense(32, kernel_constraint=constraint)
    """

    def __init__(self, lower: float = -1.0, upper: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        """Clip weights to [lower, upper]."""
        return tf.clip_by_value(w, self.lower, self.upper)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"lower": self.lower, "upper": self.upper})
        return config
