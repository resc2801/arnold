# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Compact KAN architecture with configurable basis.

This module provides a more efficient KAN variant that can use any
basis function family (polynomial, RBF, spectral, etc.).
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf

from arnold.layers.core.registry import get_layer_class

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold")
class CompactKAN(tfk.Model):
    r"""
    Compact KAN architecture with configurable basis functions.

    A more efficient KAN variant that uses any basis function family
    from the ARNOLD registry. This provides flexibility in choosing
    the most appropriate basis for the problem domain.

    Parameters
    ----------
    layer_dims : Sequence[int]
        Dimensions for each layer, e.g., [input_dim, hidden1, hidden2, output_dim].
    basis : str, default='legendre'
        Basis function type (e.g., 'legendre', 'chebyshev1', 'fourier', 'gaussian_rbf').
    degree : int, default=5
        Polynomial degree or equivalent for the basis.
    basis_kwargs : dict | None, optional
        Additional kwargs passed to each basis layer.
    activation : str | None, optional
        Activation function between layers.
    use_residual : bool, default=False
        Whether to use residual connections (when dimensions match).

    Examples
    --------
    >>> # Legendre-based compact KAN
    >>> model = CompactKAN(
    ...     layer_dims=[2, 64, 64, 1],
    ...     basis='legendre',
    ...     degree=8
    ... )
    >>>
    >>> # Fourier-based KAN for periodic functions
    >>> model = CompactKAN(
    ...     layer_dims=[1, 32, 1],
    ...     basis='fourier',
    ...     degree=10
    ... )
    """

    def __init__(
        self,
        layer_dims: Sequence[int],
        basis: str = "legendre",
        degree: int = 5,
        basis_kwargs: dict | None = None,
        activation: str | None = None,
        use_residual: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if len(layer_dims) < 2:
            raise ValueError(
                f"layer_dims must have at least 2 elements, got {len(layer_dims)}"
            )

        self._layer_dims = list(layer_dims)
        self._basis_name = basis
        self.degree = degree
        self._basis_kwargs = basis_kwargs or {}
        self.activation_name = activation
        self.use_residual = use_residual

        self._activation = tfk.activations.get(activation)
        self._layers = []

        # Get basis class
        basis_cls = get_layer_class(basis)

        # Create layers
        for i in range(len(layer_dims) - 1):
            layer = basis_cls(
                units=layer_dims[i + 1],
                degree=degree,
                **self._basis_kwargs,
            )
            self._layers.append(layer)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass through all layers."""
        x = inputs
        for i, layer in enumerate(self._layers):
            residual = x

            x = layer(x, training=training)

            # Apply residual connection if dimensions match
            if self.use_residual and residual.shape[-1] == x.shape[-1]:
                x = x + residual

            # Apply activation between layers (not after last)
            if self._activation is not None and i < len(self._layers) - 1:
                x = self._activation(x)

        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "layer_dims": self._layer_dims,
                "basis": self._basis_name,
                "degree": self.degree,
                "basis_kwargs": self._basis_kwargs,
                "activation": self.activation_name,
                "use_residual": self.use_residual,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "CompactKAN":
        return cls(**config)
