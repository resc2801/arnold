# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Smoothness regularizer for KAN weights.

Penalizes rapid variations in weight sequences using finite differences.
"""

from __future__ import annotations

import tensorflow as tf

from arnold.layers.regularizers.base import KANRegularizer

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold")
class SmoothnessRegularizer(KANRegularizer):
    r"""
    Finite-difference smoothness regularizer.

    Penalizes the sum of squared first differences along a specified axis:

    .. math::

        R(w) = \lambda \sum_i (w_{i+1} - w_i)^2

    Encourages smooth weight sequences, useful for:

    - B-spline control points
    - Polynomial coefficient sequences
    - Any weights that should vary smoothly

    Parameters
    ----------
    weight : float, default=0.01
        Regularization strength.
    axis : int, default=-1
        Axis along which to compute differences.
    order : int, default=1
        Order of finite differences (1 or 2).
        - 1: First-order differences (velocity).
        - 2: Second-order differences (acceleration).

    Examples
    --------
    >>> # Smooth B-spline control points
    >>> reg = SmoothnessRegularizer(weight=0.1, axis=-1)
    >>> penalty = reg(spline_coeffs)
    >>>
    >>> # Penalize acceleration (curvature-like)
    >>> reg2 = SmoothnessRegularizer(weight=0.05, order=2)
    """

    def __init__(
        self, weight: float = 0.01, axis: int = -1, order: int = 1, **kwargs
    ):
        super().__init__(**kwargs)
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {order}")
        self.weight = float(weight)
        self.axis = axis
        self.order = order

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """Compute smoothness penalty via finite differences."""
        diffs = tf.experimental.numpy.diff(x, n=1, axis=self.axis)
        if self.order == 2:
            diffs = tf.experimental.numpy.diff(diffs, n=1, axis=self.axis)

        return self.weight * tf.reduce_sum(tf.square(diffs))

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {"weight": self.weight, "axis": self.axis, "order": self.order}
        )
        return config
