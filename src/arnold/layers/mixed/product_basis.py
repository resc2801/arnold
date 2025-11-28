# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Product basis layer for tensor product expansions.

This module provides a layer that computes tensor products of basis functions,
useful for multivariate function approximation.
"""

from __future__ import annotations

import tensorflow as tf

from arnold.layers.core.kan_base import KANBase
from arnold.layers.core.registry import get_layer_class


tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold")
class ProductBasis(KANBase):
    r"""
    Tensor product basis layer.

    Computes the tensor product of two basis expansions, useful for
    multivariate function approximation where the input can be split
    into distinct coordinate groups.

    For 2D input :math:`(x, y)` with bases :math:`\phi_i(x)` and :math:`\psi_j(y)`:

    .. math::

        f(x, y) = \sum_{i,j} c_{ij} \phi_i(x) \psi_j(y)

    The product basis has :math:`n \times m` terms where :math:`n` and :math:`m`
    are the degrees of the two bases.

    Parameters
    ----------
    units : int
        Output dimension.
    basis_x : str
        Basis type for first coordinate(s) (e.g., 'legendre').
    basis_y : str
        Basis type for second coordinate(s) (e.g., 'fourier').
    basis_x_kwargs : dict | None, optional
        Configuration for x-basis (e.g., {'degree': 10}).
    basis_y_kwargs : dict | None, optional
        Configuration for y-basis.
    split_dim : int, default=-1
        Dimension along which to split input into x and y parts.
        Default splits the last dimension in half.
    activation : str | None, optional
        Activation function to apply to the output.
    use_bias : bool, default=True
        Whether to include a bias term.

    Examples
    --------
    >>> # Legendre × Fourier for 2D functions
    >>> layer = ProductBasis(
    ...     units=32,
    ...     basis_x='legendre',
    ...     basis_y='fourier',
    ...     basis_x_kwargs={'degree': 8},
    ...     basis_y_kwargs={'degree': 5}
    ... )
    >>> # Input shape: (batch, 2) -> split into x and y
    >>> output = layer(tf.random.normal((16, 2)))

    Notes
    -----
    For high-dimensional inputs, consider using TensorTrain decomposition
    instead (see experimental/tensor_network_bases.py).
    """

    def __init__(
        self,
        units: int,
        basis_x: str,
        basis_y: str,
        basis_x_kwargs: dict | None = None,
        basis_y_kwargs: dict | None = None,
        split_dim: int = -1,
        activation: str | None = None,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(units=units, **kwargs)
        self._basis_x_name = basis_x
        self._basis_y_name = basis_y
        self._basis_x_kwargs = basis_x_kwargs or {}
        self._basis_y_kwargs = basis_y_kwargs or {}
        self.split_dim = split_dim
        self.activation_name = activation
        self.use_bias = use_bias

        self._activation = tfk.activations.get(activation)
        self._layer_x: KANBase | None = None
        self._layer_y: KANBase | None = None

    def build(self, input_shape):
        """Build sub-layers for x and y bases."""
        if self.built:
            return

        # Infer split point
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Input dimension must be known at build time")
        self._split_point = input_dim // 2

        # Build x-basis layer
        basis_x_cls = get_layer_class(self._basis_x_name)
        # Use intermediate units that will be combined
        x_kwargs = dict(self._basis_x_kwargs)
        degree_x = x_kwargs.pop("degree", 5)
        self._layer_x = basis_x_cls(
            units=degree_x + 1, degree=degree_x, **x_kwargs
        )
        x_shape = list(input_shape)
        x_shape[-1] = self._split_point
        self._layer_x.build(tuple(x_shape))

        # Build y-basis layer
        basis_y_cls = get_layer_class(self._basis_y_name)
        y_kwargs = dict(self._basis_y_kwargs)
        degree_y = y_kwargs.pop("degree", 5)
        self._layer_y = basis_y_cls(
            units=degree_y + 1, degree=degree_y, **y_kwargs
        )
        y_shape = list(input_shape)
        y_shape[-1] = input_dim - self._split_point
        self._layer_y.build(tuple(y_shape))

        # Product dimension
        self._product_dim = (degree_x + 1) * (degree_y + 1)

        # Output projection weights
        self._output_kernel = self.add_weight(
            name="output_kernel",
            shape=(self._product_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )

        if self.use_bias:
            self._bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
            )

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Compute tensor product of basis expansions."""
        # Split input
        x = inputs[..., : self._split_point]
        y = inputs[..., self._split_point :]

        # Evaluate both bases: (batch, n) and (batch, m)
        phi_x = self._layer_x(x, training=training)  # (batch, degree_x+1)
        psi_y = self._layer_y(y, training=training)  # (batch, degree_y+1)

        # Compute outer product: (batch, n, m)
        product = tf.einsum("...i,...j->...ij", phi_x, psi_y)

        # Flatten: (batch, n*m)
        product_flat = tf.reshape(product, [-1, self._product_dim])

        # Project to output: (batch, units)
        result = tf.matmul(product_flat, self._output_kernel)

        if self.use_bias:
            result = result + self._bias

        if self._activation is not None:
            result = self._activation(result)

        return result

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "basis_x": self._basis_x_name,
                "basis_y": self._basis_y_name,
                "basis_x_kwargs": self._basis_x_kwargs,
                "basis_y_kwargs": self._basis_y_kwargs,
                "split_dim": self.split_dim,
                "activation": self.activation_name,
                "use_bias": self.use_bias,
            }
        )
        return config
