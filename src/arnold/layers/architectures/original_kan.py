# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Original KAN architecture using B-splines.

This module implements the original KAN architecture from Liu et al. (2024)
using B-spline basis functions.
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf

from arnold.layers.core.splines import BSpline

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold")
class OriginalKAN(tfk.Model):
    r"""
    Original KAN architecture from Liu et al. (2024).

    Implements the Kolmogorov-Arnold Network using B-spline basis functions
    as described in the original paper "KAN: Kolmogorov-Arnold Networks".

    The architecture consists of multiple layers, each computing:

    .. math::

        y_j = \sum_{i} \phi_{ij}(x_i)

    where :math:`\phi_{ij}` are learnable univariate functions represented
    as B-spline expansions.

    Parameters
    ----------
    layer_dims : Sequence[int]
        Dimensions for each layer, e.g., [input_dim, hidden1, hidden2, output_dim].
    spline_order : int, default=3
        B-spline order (degree + 1). Order 3 = quadratic, 4 = cubic.
    grid_size : int, default=5
        Number of grid intervals for the B-spline basis.
    grid_range : tuple[float, float], default=(-1, 1)
        Range of the spline grid.
    activation : str | None, optional
        Activation function between layers.

    Examples
    --------
    >>> # 3-layer KAN for regression
    >>> model = OriginalKAN(
    ...     layer_dims=[2, 32, 32, 1],
    ...     spline_order=4,
    ...     grid_size=8
    ... )
    >>> model.compile(optimizer='adam', loss='mse')
    >>> model.fit(X_train, y_train, epochs=100)

    References
    ----------
    .. [1] Liu, Z., et al. "KAN: Kolmogorov-Arnold Networks." arXiv:2404.19756 (2024).
    """

    def __init__(
        self,
        layer_dims: Sequence[int],
        spline_order: int = 3,
        grid_size: int = 5,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        activation: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if len(layer_dims) < 2:
            raise ValueError(
                f"layer_dims must have at least 2 elements, got {len(layer_dims)}"
            )

        self._layer_dims = list(layer_dims)
        self.spline_order = spline_order
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.activation_name = activation

        self._activation = tfk.activations.get(activation)
        self._layers: list[BSpline] = []

        # Create BSpline layers
        for i in range(len(layer_dims) - 1):
            layer = BSpline(
                units=layer_dims[i + 1],
                degree=spline_order - 1,  # order = degree + 1
                num_knots=grid_size + spline_order,
            )
            self._layers.append(layer)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass through all KAN layers."""
        x = inputs
        for i, layer in enumerate(self._layers):
            x = layer(x, training=training)
            # Apply activation between layers (not after last)
            if self._activation is not None and i < len(self._layers) - 1:
                x = self._activation(x)
        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "layer_dims": self._layer_dims,
                "spline_order": self.spline_order,
                "grid_size": self.grid_size,
                "grid_range": self.grid_range,
                "activation": self.activation_name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "OriginalKAN":
        return cls(**config)
