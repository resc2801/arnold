# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Base class for ARNOLD regularizers.

Provides the abstract base class for all KAN-specific regularizers.
"""

from __future__ import annotations

from abc import abstractmethod

import tensorflow as tf

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold")
class KANRegularizer(tfk.regularizers.Regularizer):
    r"""
    Abstract base class for all ARNOLD/KAN regularizers.

    Extends :class:`tf.keras.regularizers.Regularizer` with KAN-specific
    conventions for documentation and configuration.

    Subclasses must implement:

    - ``__call__(self, x)``: Compute regularization penalty.
    - ``get_config(self)``: Return configuration dict.

    Notes
    -----
    All regularizer penalties should be non-negative scalars that will be
    added to the total loss during training.
    """

    @abstractmethod
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the regularization penalty.

        Parameters
        ----------
        x : tf.Tensor
            The weight tensor to regularize.

        Returns
        -------
        tf.Tensor
            Scalar penalty value (non-negative).
        """
        raise NotImplementedError

    def get_config(self) -> dict:
        """Return configuration for serialization."""
        return {}

    @classmethod
    def from_config(cls, config: dict) -> "KANRegularizer":
        """Reconstruct from configuration."""
        return cls(**config)
