# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Monotonicity constraint for KAN weights.

This module provides constraints ensuring weight sequences are monotonic.
"""

from __future__ import annotations

import tensorflow as tf

from arnold.layers.constraints.base import KANConstraint

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold")
class MonotonicityConstraint(KANConstraint):
    r"""
    Keras constraint ensuring weights form a monotonic sequence.

    For a weight vector :math:`w = [w_0, w_1, \ldots, w_n]`, enforces either:

    - **Increasing**: :math:`w_0 \leq w_1 \leq \ldots \leq w_n`
    - **Decreasing**: :math:`w_0 \geq w_1 \geq \ldots \geq w_n`

    This is achieved by representing weights as cumulative sums of
    non-negative differences.

    Parameters
    ----------
    increasing : bool, default=True
        If True, enforce increasing monotonicity.
        If False, enforce decreasing monotonicity.
    axis : int, default=-1
        The axis along which to enforce monotonicity.

    Examples
    --------
    >>> # For spline control points that should be increasing:
    >>> constraint = MonotonicityConstraint(increasing=True)
    >>> 
    >>> # For decay rates that should be decreasing:
    >>> constraint = MonotonicityConstraint(increasing=False)
    """

    def __init__(self, increasing: bool = True, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.increasing = increasing
        self.axis = axis

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        """Apply monotonicity constraint via cumulative softplus differences."""
        # Get the first element
        first = tf.gather(w, [0], axis=self.axis)

        # Compute differences and ensure they're non-negative
        diffs = tf.experimental.numpy.diff(w, axis=self.axis)
        diffs = tf.nn.softplus(diffs)  # Ensure non-negative differences

        # Reconstruct via cumsum
        if self.increasing:
            cumsum = tf.cumsum(diffs, axis=self.axis)
            result = tf.concat([first, first + cumsum], axis=self.axis)
        else:
            cumsum = tf.cumsum(diffs, axis=self.axis)
            result = tf.concat([first, first - cumsum], axis=self.axis)

        return result

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"increasing": self.increasing, "axis": self.axis})
        return config
