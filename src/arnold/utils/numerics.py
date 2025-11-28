## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""
Numerically stable helper functions for KAN layers.
"""

import tensorflow as tf


def safe_acos(x: tf.Tensor, eps: float = 1e-7) -> tf.Tensor:
    """
    Numerically stable arccos on ``[-1, 1]`` with clipping to avoid NaNs.
    """
    return tf.math.acos(tf.clip_by_value(x, -1.0 + eps, 1.0 - eps))


def safe_log(x: tf.Tensor, eps: float = 1e-7) -> tf.Tensor:
    """
    Numerically stable logarithm with floor to avoid ``-inf`` at zero.
    """
    return tf.math.log(tf.maximum(x, eps))


def safe_reciprocal(x: tf.Tensor, eps: float = 1e-7) -> tf.Tensor:
    """
    Numerically stable reciprocal avoiding division by zero.
    """
    return x / (tf.square(x) + eps)


def clamp_abs(x: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
    """
    Clamp values away from zero by ``eps`` preserving sign, useful for avoiding poles.

    For values with ``abs(x) < eps``:

    - Positive or zero values become +eps
    - Negative values become -eps
    """
    eps_t = tf.cast(eps, x.dtype)
    # sign(0) = 0, so we need special handling for zero
    # Use where to map: x >= 0 -> eps, x < 0 -> -eps
    clamped = tf.where(x >= 0, eps_t, -eps_t)
    return tf.where(tf.abs(x) < eps_t, clamped, x)
