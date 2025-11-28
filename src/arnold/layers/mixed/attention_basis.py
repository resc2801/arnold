# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Attention-weighted basis combination layer.

This module provides a layer that uses attention mechanisms to dynamically
weight contributions from multiple basis functions.
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf

from arnold.layers.core.kan_base import KANBase
from arnold.layers.core.registry import get_layer_class

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold")
class AttentionBasis(KANBase):
    r"""
    Attention-weighted basis combination layer.

    Uses a learned attention mechanism to dynamically weight contributions
    from multiple basis function families based on the input.

    For input :math:`x` and bases :math:`\phi_1, \ldots, \phi_K`:

    .. math::

        f(x) = \sum_{k=1}^{K} \text{softmax}(q(x)^\top k_k) \cdot v_k(x)

    where :math:`q(x)` is a query projection of the input, :math:`k_k` are
    learned key vectors, and :math:`v_k(x) = \phi_k(x) \cdot w_k` are
    value projections from each basis.

    Parameters
    ----------
    units : int
        Output dimension.
    bases : Sequence[str]
        List of basis names (e.g., ['legendre', 'fourier', 'gaussian_rbf']).
    basis_kwargs : dict[str, dict] | None, optional
        Per-basis configuration. Keys are basis names, values are kwargs dicts.
    attention_dim : int, default=32
        Dimension of the attention query/key space.
    temperature : float, default=1.0
        Softmax temperature (lower = sharper attention).
    activation : str | None, optional
        Activation function to apply to the output.
    use_bias : bool, default=True
        Whether to include a bias term.

    Examples
    --------
    >>> # Attention over multiple bases
    >>> layer = AttentionBasis(
    ...     units=32,
    ...     bases=['legendre', 'fourier', 'gaussian_rbf'],
    ...     attention_dim=16,
    ...     temperature=0.5
    ... )
    """

    def __init__(
        self,
        units: int,
        bases: Sequence[str],
        basis_kwargs: dict[str, dict] | None = None,
        attention_dim: int = 32,
        temperature: float = 1.0,
        activation: str | None = None,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(units=units, **kwargs)
        if len(bases) < 2:
            raise ValueError(
                f"AttentionBasis requires at least 2 bases, got {len(bases)}"
            )
        self._bases_names = list(bases)
        self._basis_kwargs = basis_kwargs or {}
        self.attention_dim = attention_dim
        self.temperature = temperature
        self.activation_name = activation
        self.use_bias = use_bias

        self._activation = tfk.activations.get(activation)
        self._sub_layers: list[KANBase] = []

    def build(self, input_shape):
        """Build sub-layers and attention mechanism."""
        if self.built:
            return

        input_dim = input_shape[-1]
        num_bases = len(self._bases_names)

        # Create sub-layers for each basis
        for basis_name in self._bases_names:
            basis_cls = get_layer_class(basis_name)
            basis_kw = self._basis_kwargs.get(basis_name, {})
            layer = basis_cls(units=self.units, **basis_kw)
            layer.build(input_shape)
            self._sub_layers.append(layer)

        # Query projection: input -> attention_dim
        self._query_proj = self.add_weight(
            name="query_proj",
            shape=(input_dim, self.attention_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Key vectors: one per basis
        self._keys = self.add_weight(
            name="keys",
            shape=(num_bases, self.attention_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Optional bias
        if self.use_bias:
            self._bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
            )

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass with attention-weighted basis combination."""
        # Compute query: (batch, attention_dim)
        query = tf.matmul(inputs, self._query_proj)

        # Compute attention scores: (batch, num_bases)
        # query: (batch, attention_dim), keys: (num_bases, attention_dim)
        scores = tf.matmul(query, self._keys, transpose_b=True)
        scores = scores / self.temperature

        # Softmax attention weights: (batch, num_bases)
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # Compute weighted sum of basis outputs
        # Each basis output: (batch, units)
        outputs = []
        for layer in self._sub_layers:
            basis_output = layer(inputs, training=training)
            outputs.append(basis_output)

        # Stack outputs: (batch, num_bases, units)
        stacked = tf.stack(outputs, axis=1)

        # Apply attention: (batch, num_bases, 1) * (batch, num_bases, units)
        attention_weights = tf.expand_dims(attention_weights, axis=-1)
        weighted = attention_weights * stacked

        # Sum over bases: (batch, units)
        result = tf.reduce_sum(weighted, axis=1)

        # Add bias
        if self.use_bias:
            result = result + self._bias

        # Apply activation
        if self._activation is not None:
            result = self._activation(result)

        return result

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "bases": self._bases_names,
                "basis_kwargs": self._basis_kwargs,
                "attention_dim": self.attention_dim,
                "temperature": self.temperature,
                "activation": self.activation_name,
                "use_bias": self.use_bias,
            }
        )
        return config

    def get_attention_weights(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Get attention weights for given inputs.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Attention weights of shape (batch, num_bases).
        """
        query = tf.matmul(inputs, self._query_proj)
        scores = tf.matmul(query, self._keys, transpose_b=True)
        scores = scores / self.temperature
        return tf.nn.softmax(scores, axis=-1)
