# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Positivity constraint for KAN weights.

This module provides constraints ensuring weights remain strictly positive.
"""

from __future__ import annotations

import tensorflow as tf

from arnold.layers.constraints.base import KANConstraint


tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold")
class PositivityConstraint(KANConstraint):
    r"""
    Keras constraint ensuring weights are strictly positive.

    Uses softplus transformation to map any values to :math:`(\epsilon, \infty)`:

    .. math::

        f(w) = \text{softplus}(w) + \epsilon

    This is differentiable everywhere, unlike ``tf.maximum(w, eps)`` which
    has zero gradients for :math:`w < \epsilon`.

    Parameters
    ----------
    eps : float, default=1e-6
        Small positive offset to ensure strict positivity.
    mode : str, default="softplus"
        Transformation mode:
        - "softplus": Use softplus transformation (differentiable)
        - "clip": Use clipping (non-differentiable at boundary)
        - "abs": Use absolute value (non-differentiable at zero)

    Examples
    --------
    >>> constraint = PositivityConstraint()
    >>> layer = tf.keras.layers.Dense(32, kernel_constraint=constraint)
    >>>
    >>> # For RBF scale parameters:
    >>> constraint = PositivityConstraint(eps=1e-4, mode="softplus")
    """

    def __init__(self, eps: float = 1e-6, mode: str = "softplus", **kwargs):
        super().__init__(**kwargs)
        if mode not in ("softplus", "clip", "abs"):
            raise ValueError(f"mode must be 'softplus', 'clip', or 'abs', got {mode}")
        self.eps = eps
        self.mode = mode

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        """Apply positivity constraint."""
        if self.mode == "softplus":
            return tf.nn.softplus(w) + self.eps
        elif self.mode == "clip":
            return tf.maximum(w, self.eps)
        else:  # mode == "abs"
            return tf.abs(w) + self.eps

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"eps": self.eps, "mode": self.mode})
        return config
