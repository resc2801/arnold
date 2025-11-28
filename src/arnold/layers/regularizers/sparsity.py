# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Sparsity regularizer for KAN weights.

Promotes sparse activation patterns via entropy-based penalties.
"""

from __future__ import annotations

import tensorflow as tf

from arnold.layers.regularizers.base import KANRegularizer

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold")
class SparsityRegularizer(KANRegularizer):
    r"""
    Entropy-based sparsity regularizer.

    Encourages activation sparsity by penalizing deviations from a target
    average activation level. Based on the Kullback-Leibler divergence
    between the target and actual average activations.

    For average activation :math:`\hat{\rho}` and target :math:`\rho`:

    .. math::

        R = \rho \log\frac{\rho}{\hat{\rho}} +
            (1-\rho) \log\frac{1-\rho}{1-\hat{\rho}}

    Parameters
    ----------
    target_sparsity : float, default=0.05
        Target average activation level (0 to 1).
        Lower values promote more sparsity.
    weight : float, default=1.0
        Scaling factor for the penalty.

    Notes
    -----
    This regularizer is designed for activation-like values in [0, 1].
    For weight matrices, consider using L1 regularization instead.

    Examples
    --------
    >>> reg = SparsityRegularizer(target_sparsity=0.05, weight=3.0)
    >>> penalty = reg(hidden_activations)
    """

    def __init__(
        self, target_sparsity: float = 0.05, weight: float = 1.0, **kwargs
    ):
        super().__init__(**kwargs)
        if not 0 < target_sparsity < 1:
            raise ValueError(
                f"target_sparsity must be in (0, 1), got {target_sparsity}"
            )
        self.target_sparsity = float(target_sparsity)
        self.weight = float(weight)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """Compute KL-divergence based sparsity penalty."""
        # Compute average activation
        rho_hat = tf.reduce_mean(tf.abs(x))

        # Clip to avoid log(0)
        eps = 1e-10
        rho_hat = tf.clip_by_value(rho_hat, eps, 1.0 - eps)

        rho = self.target_sparsity

        # KL divergence
        kl = rho * tf.math.log(rho / rho_hat) + (1 - rho) * tf.math.log(
            (1 - rho) / (1 - rho_hat)
        )

        return self.weight * kl

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {"target_sparsity": self.target_sparsity, "weight": self.weight}
        )
        return config
