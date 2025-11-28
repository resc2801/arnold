# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Kalman KAN architecture with recursive filtering.

This module provides a KAN variant that incorporates Kalman filter-like
recursive state updates for sequential/time-series data.
"""

from __future__ import annotations

import tensorflow as tf

from arnold.layers.core.registry import get_layer_class


tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold")
class KalmanKAN(tfk.Model):
    r"""
    Kalman KAN architecture with recursive state filtering.

    Combines KAN basis expansions with Kalman filter-inspired state
    updates for sequential data processing. The state evolves as:

    .. math::

        h_t = A \cdot h_{t-1} + B \cdot \phi(x_t)
        y_t = C \cdot h_t

    where :math:`\phi(x_t)` is the KAN basis expansion and :math:`A, B, C`
    are learned transition/observation matrices.

    Parameters
    ----------
    state_dim : int
        Dimension of the hidden state.
    output_dim : int
        Output dimension.
    basis : str, default='legendre'
        Basis function type for input expansion.
    degree : int, default=5
        Polynomial degree for the basis.
    basis_kwargs : dict | None, optional
        Additional kwargs for the basis layer.
    use_gate : bool, default=True
        Whether to use gating mechanism (similar to GRU/LSTM).

    Examples
    --------
    >>> # Kalman KAN for time series
    >>> model = KalmanKAN(
    ...     state_dim=64,
    ...     output_dim=1,
    ...     basis='fourier',
    ...     degree=8
    ... )
    >>> # Process sequence
    >>> outputs = []
    >>> state = None
    >>> for t in range(seq_len):
    ...     y, state = model(x[:, t:t+1], state)
    ...     outputs.append(y)
    """

    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        basis: str = "legendre",
        degree: int = 5,
        basis_kwargs: dict | None = None,
        use_gate: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.output_dim = output_dim
        self._basis_name = basis
        self.degree = degree
        self._basis_kwargs = basis_kwargs or {}
        self.use_gate = use_gate

        # Basis layer for input expansion
        basis_cls = get_layer_class(basis)
        self._basis_layer = basis_cls(
            units=state_dim,
            degree=degree,
            **self._basis_kwargs,
        )

        # State transition (A matrix)
        self._transition = tfk.layers.Dense(state_dim, use_bias=False)

        # Input projection (B matrix)
        self._input_proj = tfk.layers.Dense(state_dim, use_bias=True)

        # Output projection (C matrix)
        self._output_proj = tfk.layers.Dense(output_dim, use_bias=True)

        # Optional gating mechanism
        if use_gate:
            self._gate = tfk.layers.Dense(state_dim, activation="sigmoid")

    def call(
        self,
        inputs: tf.Tensor,
        states: tf.Tensor | None = None,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass with state update.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch, input_dim).
        states : tf.Tensor | None
            Previous state of shape (batch, state_dim).
            If None, initializes with zeros.
        training : bool
            Whether in training mode.

        Returns
        -------
        output : tf.Tensor
            Output tensor of shape (batch, output_dim).
        new_state : tf.Tensor
            Updated state of shape (batch, state_dim).
        """
        batch_size = tf.shape(inputs)[0]

        # Initialize state if needed
        if states is None:
            states = tf.zeros((batch_size, self.state_dim), dtype=inputs.dtype)

        # Expand input with basis functions
        phi = self._basis_layer(inputs, training=training)

        # Compute state transition
        state_trans = self._transition(states)
        input_update = self._input_proj(phi)

        if self.use_gate:
            # Gated update (GRU-like)
            gate = self._gate(tf.concat([states, phi], axis=-1))
            new_state = gate * state_trans + (1 - gate) * input_update
        else:
            # Simple additive update
            new_state = state_trans + input_update

        # Compute output
        output = self._output_proj(new_state)

        return output, new_state

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "state_dim": self.state_dim,
                "output_dim": self.output_dim,
                "basis": self._basis_name,
                "degree": self.degree,
                "basis_kwargs": self._basis_kwargs,
                "use_gate": self.use_gate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> KalmanKAN:
        return cls(**config)
