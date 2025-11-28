# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Base constraint class for KAN layers.

This module provides the abstract base class for all KAN-specific constraints.
"""

from abc import abstractmethod

import tensorflow as tf

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold")
class KANConstraint(tfk.constraints.Constraint):
    r"""
    Abstract base class for KAN-specific weight constraints.

    KAN constraints enforce mathematical requirements on layer weights
    while maintaining smooth gradient flow. Unlike hard constraints
    (e.g., ``tf.maximum``), these use differentiable transformations.

    Subclasses must implement:

    - :meth:`__call__` — Apply the constraint transformation

    Notes
    -----
    All constraints are Keras-serializable and can be saved/loaded
    with models.
    """

    @abstractmethod
    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        r"""
        Apply the constraint to the weight tensor.

        Parameters
        ----------
        w : tf.Tensor
            Weight tensor to constrain.

        Returns
        -------
        tf.Tensor
            Constrained weight tensor with the same shape.
        """
        raise NotImplementedError

    def get_config(self) -> dict:
        """Return configuration for serialization."""
        return {}
