# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Mixed basis layer combining multiple basis function families.

This module provides a layer that combines outputs from multiple different
basis function types with learned mixing weights.
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf

from arnold.layers.core.kan_base import KANBase
from arnold.layers.core.registry import get_layer_class

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold")
class MixedBasis(KANBase):
    r"""
    Mixed basis layer combining multiple basis function families.

    This layer creates multiple sub-basis layers and combines their outputs
    with learned weights. Useful for representing functions that have
    different characteristics in different regions or scales.

    Given basis functions :math:`\phi_1, \phi_2, \ldots, \phi_K` from different
    families, the output is:

    .. math::

        f(x) = \sum_{k=1}^{K} \alpha_k \cdot \phi_k(x) \cdot w_k

    where :math:`\alpha_k` are learned mixing weights (optionally softmax-normalized).

    Parameters
    ----------
    units : int
        Output dimension.
    bases : Sequence[str]
        List of basis names (e.g., ['legendre', 'fourier', 'gaussian_rbf']).
    basis_kwargs : dict[str, dict] | None, optional
        Per-basis configuration. Keys are basis names, values are kwargs dicts.
    normalize_weights : bool, default=True
        If True, apply softmax to mixing weights so they sum to 1.
    activation : str | None, optional
        Activation function to apply to the output.
    use_bias : bool, default=True
        Whether to include a bias term.

    Examples
    --------
    >>> # Combine Legendre and Fourier bases
    >>> layer = MixedBasis(
    ...     units=32,
    ...     bases=['legendre', 'fourier'],
    ...     basis_kwargs={'legendre': {'degree': 10}, 'fourier': {'degree': 5}}
    ... )
    >>>
    >>> # Three-way mix with RBF
    >>> layer = MixedBasis(
    ...     units=64,
    ...     bases=['chebyshev1', 'gaussian_rbf', 'rff'],
    ...     normalize_weights=True
    ... )
    """

    def __init__(
        self,
        units: int,
        bases: Sequence[str],
        basis_kwargs: dict[str, dict] | None = None,
        normalize_weights: bool = True,
        activation: str | None = None,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(units=units, **kwargs)
        if len(bases) < 2:
            raise ValueError(
                f"MixedBasis requires at least 2 bases, got {len(bases)}"
            )
        self._bases_names = list(bases)
        self._basis_kwargs = basis_kwargs or {}
        self.normalize_weights = normalize_weights
        self.activation_name = activation
        self.use_bias = use_bias

        self._activation = tfk.activations.get(activation)
        self._sub_layers: list[KANBase] = []

    def build(self, input_shape):
        """Build sub-layers and mixing weights."""
        if self.built:
            return

        # Create sub-layers for each basis
        for basis_name in self._bases_names:
            basis_cls = get_layer_class(basis_name)
            basis_kw = self._basis_kwargs.get(basis_name, {})
            # Create layer with same units
            layer = basis_cls(units=self.units, **basis_kw)
            layer.build(input_shape)
            self._sub_layers.append(layer)
            # Track sub-layer weights
            for weight in layer.trainable_weights:
                self._trainable_weights.append(weight)

        # Create mixing weights (one per basis)
        num_bases = len(self._bases_names)
        self._mixing_logits = self.add_weight(
            name="mixing_logits",
            shape=(num_bases,),
            initializer="zeros",
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
        """Forward pass combining all basis outputs."""
        # Compute mixing weights
        if self.normalize_weights:
            mixing_weights = tf.nn.softmax(self._mixing_logits)
        else:
            mixing_weights = tf.nn.sigmoid(self._mixing_logits)

        # Compute weighted sum of basis outputs
        outputs = []
        for i, layer in enumerate(self._sub_layers):
            basis_output = layer(inputs, training=training)
            weighted = mixing_weights[i] * basis_output
            outputs.append(weighted)

        result = tf.add_n(outputs)

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
                "normalize_weights": self.normalize_weights,
                "activation": self.activation_name,
                "use_bias": self.use_bias,
            }
        )
        return config

    @property
    def mixing_weights(self) -> tf.Tensor:
        """Get current mixing weights (normalized if applicable)."""
        if self.normalize_weights:
            return tf.nn.softmax(self._mixing_logits)
        else:
            return tf.nn.sigmoid(self._mixing_logits)
