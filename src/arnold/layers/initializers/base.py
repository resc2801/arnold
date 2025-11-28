# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Base class and utility functions for ARNOLD initializers.

Provides the abstract base class and helper functions for creating
trainable parameters with optional constraints.
"""

from __future__ import annotations

from abc import abstractmethod

import tensorflow as tf

from arnold.layers.constraints import inverse_softplus

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold")
class KANInitializer(tfk.initializers.Initializer):
    r"""
    Abstract base class for all ARNOLD/KAN initializers.

    Extends :class:`tf.keras.initializers.Initializer` with KAN-specific
    conventions for documentation and configuration.

    Subclasses must implement:

    - ``__call__(self, shape, dtype=None)``: Generate initial values.
    - ``get_config(self)``: Return configuration dict.
    """

    @abstractmethod
    def __call__(self, shape, dtype=None) -> tf.Tensor:
        """
        Generate initial weight values.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the weight tensor.
        dtype : tf.DType, optional
            Data type for the weights.

        Returns
        -------
        tf.Tensor
            Initial weight values.
        """
        raise NotImplementedError

    def get_config(self) -> dict:
        """Return configuration for serialization."""
        return {}

    @classmethod
    def from_config(cls, config: dict) -> "KANInitializer":
        """Reconstruct from configuration."""
        return cls(**config)


def create_trainable_param(
    layer: tf.keras.layers.Layer,
    name: str,
    init_value: float | None = None,
    *,
    initializer: str | tf.keras.initializers.Initializer | None = None,
    trainable: bool = True,
    constraint: tf.keras.constraints.Constraint | None = None,
) -> tf.Variable:
    r"""
    Create a scalar trainable parameter with optional constant initialization.

    Parameters
    ----------
    layer : tf.keras.layers.Layer
        Owning layer (used to register the weight).
    name : str
        Weight name.
    init_value : float | None, optional
        Constant initialization when provided.
    initializer : str | tf.keras.initializers.Initializer | None, optional
        Override initializer; when omitted and ``init_value`` is None,
        defaults to ``HeNormal``.
    trainable : bool, default True
        Trainability flag.
    constraint : tf.keras.constraints.Constraint | None, optional
        Optional constraint to apply.

    Returns
    -------
    tf.Variable
        The created weight variable.

    Examples
    --------
    >>> param = create_trainable_param(layer, "alpha", init_value=1.0)
    """
    if initializer is not None:
        init = tfk.initializers.get(initializer)
    elif init_value is not None:
        init = tfk.initializers.Constant(value=init_value)
    else:
        init = tfk.initializers.HeNormal()

    return layer.add_weight(
        initializer=init,
        name=name,
        trainable=trainable,
        constraint=constraint,
    )


def create_bounded_param(
    layer: tf.keras.layers.Layer,
    name: str,
    init_value: float | None = None,
    *,
    lower_bound: float,
    eps: float = 1e-6,
    trainable: bool = True,
) -> tf.Variable:
    r"""
    Create logits for a parameter constrained to :math:`(\text{lower\_bound}, \infty)`.

    The logits are initialized such that
    ``softplus_lower_bound(logits, lower_bound, eps)`` produces the target
    ``init_value``. Use with ``softplus_lower_bound()`` in the forward pass.

    Parameters
    ----------
    layer : tf.keras.layers.Layer
        Owning layer (used to register the weight).
    name : str
        Weight name (will have '_logits' appended).
    init_value : float | None, optional
        Target initial value in the constrained space.
        Must be > lower_bound + eps. Defaults to HeNormal initialization if None.
    lower_bound : float
        The exclusive lower bound for the constrained parameter.
    eps : float, default 1e-6
        Small offset from the boundary.
    trainable : bool, default True
        Trainability flag.

    Returns
    -------
    tf.Variable
        The created logits weight variable.

    Examples
    --------
    >>> # In build():
    >>> self.alpha_logits = create_bounded_param(
    ...     self, "alpha", init_value=1.0, lower_bound=-0.5
    ... )
    >>> # In forward pass:
    >>> from arnold.layers.constraints import softplus_lower_bound
    >>> alpha = softplus_lower_bound(self.alpha_logits, lower_bound=-0.5)
    """
    if init_value is not None:
        # Validate init_value is in valid range
        if init_value <= lower_bound + eps:
            raise ValueError(
                f"init_value={init_value} must be > lower_bound + eps = "
                f"{lower_bound + eps}"
            )
        # Compute logits that produce the target value
        target_logit = inverse_softplus(
            tf.constant(init_value - lower_bound - eps, dtype=tf.float32)
        )
        init = tfk.initializers.Constant(value=float(target_logit.numpy()))
    else:
        # Default to HeNormal for unconstrained logits
        init = tfk.initializers.HeNormal()

    return layer.add_weight(
        initializer=init,
        name=f"{name}_logits",
        trainable=trainable,
    )
