## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""Helpers for creating trainable scalar parameters with sane defaults."""
# ruff: isort:skip_file

from __future__ import annotations

import tensorflow as tf

from arnold.utils.constraints import inverse_softplus_lower_bound

tfk = tf.keras


def create_trainable_param(
    layer: tf.keras.layers.Layer,
    name: str,
    init_value: float | None = None,
    *,
    initializer: str | tf.keras.initializers.Initializer | None = None,
    trainable: bool = True,
    constraint: tf.keras.constraints.Constraint | None = None,
) -> tf.Variable:
    """
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
        Override initializer; when omitted and ``init_value`` is None, defaults to ``HeNormal``.
    trainable : bool, default True
        Trainability flag.
    constraint : tf.keras.constraints.Constraint | None, optional
        Optional constraint to apply.

    Returns
    -------
    tf.Variable
        The created weight variable.
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


def create_bounded_param_logits(
    layer: tf.keras.layers.Layer,
    name: str,
    init_value: float | None = None,
    *,
    lower_bound: float,
    eps: float = 1e-6,
    trainable: bool = True,
) -> tf.Variable:
    """
    Create logits for a parameter constrained to (lower_bound, ∞).

    The logits are initialized such that softplus_lower_bound(logits, lower_bound, eps)
    produces the target init_value. Use with softplus_lower_bound() in the forward pass.

    Parameters
    ----------
    layer : tf.keras.layers.Layer
        Owning layer (used to register the weight).
    name : str
        Weight name (will have '_logits' appended).
    init_value : float | None, optional
        Target initial value in the constrained space. Must be > lower_bound + eps.
        Defaults to HeNormal initialization if None.
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
    >>> self.alpha_logits = create_bounded_param_logits(
    ...     self, "alpha", init_value=1.0, lower_bound=-0.5
    ... )
    >>> # In forward pass:
    >>> alpha = softplus_lower_bound(self.alpha_logits, lower_bound=-0.5)
    """
    if init_value is not None:
        # Validate init_value is in valid range
        if init_value <= lower_bound + eps:
            raise ValueError(
                f"init_value={init_value} must be > lower_bound + eps = {lower_bound + eps}"
            )
        # Compute logits that produce the target value
        target_logit = inverse_softplus_lower_bound(
            tf.constant(init_value, dtype=tf.float32),
            lower_bound=lower_bound,
            eps=eps
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
