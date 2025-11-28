# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
L1, L2, and combined L1L2 regularizers.

Standard norm-based regularization for KAN weights.
"""

from __future__ import annotations

import tensorflow as tf

from arnold.layers.regularizers.base import KANRegularizer

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold")
class L1Regularizer(KANRegularizer):
    r"""
    L1 (Lasso) regularization penalty.

    Computes:

    .. math::

        R(w) = \lambda \sum_i |w_i|

    Promotes sparsity by driving small weights to exactly zero.

    Parameters
    ----------
    l1 : float, default=0.01
        Regularization strength.

    Examples
    --------
    >>> reg = L1Regularizer(l1=1e-4)
    >>> penalty = reg(weights)
    """

    def __init__(self, l1: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.l1 = float(l1)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return self.l1 * tf.reduce_sum(tf.abs(x))

    def get_config(self) -> dict:
        config = super().get_config()
        config["l1"] = self.l1
        return config


@tfk.utils.register_keras_serializable(package="arnold")
class L2Regularizer(KANRegularizer):
    r"""
    L2 (Ridge) regularization penalty.

    Computes:

    .. math::

        R(w) = \lambda \sum_i w_i^2

    Penalizes large weights, encouraging smaller, more distributed values.

    Parameters
    ----------
    l2 : float, default=0.01
        Regularization strength.

    Examples
    --------
    >>> reg = L2Regularizer(l2=1e-3)
    >>> penalty = reg(weights)
    """

    def __init__(self, l2: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.l2 = float(l2)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return self.l2 * tf.reduce_sum(tf.square(x))

    def get_config(self) -> dict:
        config = super().get_config()
        config["l2"] = self.l2
        return config


@tfk.utils.register_keras_serializable(package="arnold")
class L1L2Regularizer(KANRegularizer):
    r"""
    Combined L1 + L2 (Elastic Net) regularization.

    Computes:

    .. math::

        R(w) = \lambda_1 \sum_i |w_i| + \lambda_2 \sum_i w_i^2

    Combines sparsity-inducing L1 with weight-shrinking L2.

    Parameters
    ----------
    l1 : float, default=0.0
        L1 regularization strength.
    l2 : float, default=0.01
        L2 regularization strength.

    Examples
    --------
    >>> reg = L1L2Regularizer(l1=1e-4, l2=1e-3)
    >>> penalty = reg(weights)
    """

    def __init__(self, l1: float = 0.0, l2: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.l1 = float(l1)
        self.l2 = float(l2)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        penalty = tf.constant(0.0, dtype=x.dtype)
        if self.l1 > 0.0:
            penalty = penalty + self.l1 * tf.reduce_sum(tf.abs(x))
        if self.l2 > 0.0:
            penalty = penalty + self.l2 * tf.reduce_sum(tf.square(x))
        return penalty

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"l1": self.l1, "l2": self.l2})
        return config
