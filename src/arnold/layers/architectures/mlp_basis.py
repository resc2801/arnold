# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
MLP as a learnable basis function.

This module provides a layer that uses a small MLP as the basis function,
allowing fully learnable nonlinear transformations.
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf

from arnold.layers.core.kan_base import KANBase

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold")
class MLPBasis(KANBase):
    r"""
    MLP-based learnable basis function layer.

    Uses a small multi-layer perceptron as the basis function instead
    of fixed polynomial/RBF/wavelet bases. This provides maximum
    flexibility at the cost of interpretability.

    For each output unit :math:`j`, the basis function is:

    .. math::

        \phi_j(x) = \text{MLP}_j(x) = W_L \cdot \sigma(\ldots \sigma(W_1 x + b_1) \ldots) + b_L

    Parameters
    ----------
    units : int
        Output dimension.
    hidden_dims : Sequence[int], default=(32,)
        Hidden layer dimensions for the MLP basis.
    activation : str, default='gelu'
        Activation function in hidden layers.
    output_activation : str | None, optional
        Activation for the output layer.
    use_bias : bool, default=True
        Whether to use bias in MLP layers.
    dropout : float, default=0.0
        Dropout rate between hidden layers.

    Examples
    --------
    >>> # Simple MLP basis
    >>> layer = MLPBasis(
    ...     units=32,
    ...     hidden_dims=[64, 64]
    ... )
    >>>
    >>> # Deeper MLP for complex functions
    >>> layer = MLPBasis(
    ...     units=16,
    ...     hidden_dims=[128, 64, 32],
    ...     activation='swish',
    ...     dropout=0.1
    ... )

    Notes
    -----
    Unlike polynomial or RBF bases, MLP bases do not provide interpretable
    features. Use this when approximation power is more important than
    understanding the learned function structure.
    """

    def __init__(
        self,
        units: int,
        hidden_dims: Sequence[int] = (32,),
        activation: str = "gelu",
        output_activation: str | None = None,
        use_bias: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        # Remove 'degree' if passed (not applicable for MLP)
        kwargs.pop("degree", None)
        super().__init__(units=units, **kwargs)

        self._hidden_dims = list(hidden_dims)
        self.activation_name = activation
        self.output_activation_name = output_activation
        self.use_bias = use_bias
        self.dropout_rate = dropout

        # Build MLP layers
        self._hidden_layers = []
        self._dropout_layers = []

        for dim in hidden_dims:
            self._hidden_layers.append(
                tfk.layers.Dense(dim, activation=activation, use_bias=use_bias)
            )
            if dropout > 0:
                self._dropout_layers.append(tfk.layers.Dropout(dropout))
            else:
                self._dropout_layers.append(None)

        # Output layer
        self._output_layer = tfk.layers.Dense(
            units,
            activation=output_activation,
            use_bias=use_bias,
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass through the MLP."""
        x = inputs

        for hidden_layer, dropout_layer in zip(
            self._hidden_layers, self._dropout_layers
        ):
            x = hidden_layer(x)
            if dropout_layer is not None:
                x = dropout_layer(x, training=training)

        return self._output_layer(x)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "hidden_dims": self._hidden_dims,
                "activation": self.activation_name,
                "output_activation": self.output_activation_name,
                "use_bias": self.use_bias,
                "dropout": self.dropout_rate,
            }
        )
        return config
