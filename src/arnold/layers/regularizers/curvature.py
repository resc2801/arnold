# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Curvature regularizer for KAN weights.

Penalizes high curvature in learned functions via second-derivative approximations.
"""

from __future__ import annotations

import tensorflow as tf

from arnold.layers.regularizers.base import KANRegularizer

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold")
class CurvatureRegularizer(KANRegularizer):
    r"""
    Second-derivative curvature penalty regularizer.

    For weight sequence :math:`w`, approximates second derivative via
    finite differences:

    .. math::

        w''_i \approx w_{i+1} - 2w_i + w_{i-1}

    and penalizes:

    .. math::

        R(w) = \lambda \sum_i (w''_i)^2

    This encourages smooth, low-curvature functions.

    Parameters
    ----------
    weight : float, default=0.01
        Regularization strength.
    axis : int, default=-1
        Axis along which to compute curvature.

    Notes
    -----
    For polynomials, this penalizes high-degree oscillations.
    For splines, this encourages natural spline behavior (minimizing
    integrated squared curvature).

    Examples
    --------
    >>> reg = CurvatureRegularizer(weight=0.1)
    >>> penalty = reg(spline_coeffs)
    """

    def __init__(self, weight: float = 0.01, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.weight = float(weight)
        self.axis = axis

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """Compute curvature penalty via second-order finite differences."""
        # Second derivative: w[i+1] - 2*w[i] + w[i-1]
        # Equivalently, diff(diff(w))
        first_diff = tf.experimental.numpy.diff(x, n=1, axis=self.axis)
        second_diff = tf.experimental.numpy.diff(first_diff, n=1, axis=self.axis)

        return self.weight * tf.reduce_sum(tf.square(second_diff))

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"weight": self.weight, "axis": self.axis})
        return config
