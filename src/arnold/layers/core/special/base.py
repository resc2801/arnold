# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Base class for special function KAN layers.

This module provides :class:`SpecialBase`, an abstract base class for KAN
layers using special mathematical functions as basis functions.

Special functions include solutions to important differential equations
(Airy, Bessel, Mathieu, etc.) and other transcendental functions with
applications in physics and engineering.
"""
from abc import abstractmethod
from typing import Any

import tensorflow as tf

from arnold.layers.core.kan_base import KANBase


tfk = tf.keras


class SpecialBase(KANBase):
    r"""
    Abstract base class for special function KAN layers.

    Special functions are typically defined on specific domains (e.g., [0, ∞)
    for Bessel functions) and may require careful numerical handling for
    stability at boundary values or asymptotic regions.

    Subclasses must implement:

    - :meth:`_get_num_basis_functions`: Return the number of basis functions
    - :meth:`special_basis`: Compute the basis function values

    Parameters
    ----------
    units : int
        Output dimension.
    max_order : int
        Maximum order/degree of the special functions.
    input_clip : tuple[float, float] | None
        Input clipping range for numerical stability.
    use_bias : bool, default=True
        Whether to add a bias term.
    activation : str or callable, default=None
        Activation function.
    kernel_regularizer : regularizer, default=None
        Regularizer for coefficients.
    **kwargs
        Additional Keras layer arguments.

    Notes
    -----
    Special functions often require different numerical techniques than
    polynomials:

    - **Asymptotic expansions** for large arguments
    - **Power series** for small arguments
    - **Connection formulas** between different representations
    - **Recurrence relations** for function values at different orders

    Subclasses should implement these appropriately for numerical stability.
    """

    def __init__(
        self,
        units: int,
        max_order: int = 8,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        self.max_order = max_order
        super().__init__(units=units, input_clip=input_clip, **kwargs)

    @abstractmethod
    def _get_num_basis_functions(self) -> int:
        """Return the number of basis functions."""
        raise NotImplementedError

    @abstractmethod
    def special_basis(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Compute special function basis values.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, num_basis).
        """
        raise NotImplementedError

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the layer weights."""
        if self.built:
            return

        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Input dimension must be known at build time.")

        self.input_dim = input_dim
        num_basis = self._get_num_basis_functions()

        # Coefficient weights: (input_dim, num_basis, units)
        self.coefficients = self.add_weight(
            name="coefficients",
            shape=(input_dim, num_basis, self.units),
            initializer="glorot_uniform",
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype=self.dtype,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer="zeros",
                regularizer=self.bias_regularizer,
                trainable=True,
                dtype=self.dtype,
            )

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        """Forward pass through the layer."""
        x = inputs

        # Apply input clipping if specified
        if self.input_clip is not None:
            x = tf.clip_by_value(x, self.input_clip[0], self.input_clip[1])

        # Compute basis: (batch, input_dim, num_basis)
        basis = self.special_basis(x)

        # Apply coefficients: einsum over input_dim and num_basis
        # basis: (batch, input_dim, num_basis)
        # coefficients: (input_dim, num_basis, units)
        # output: (batch, units)
        output = tf.einsum("bid,idu->bu", basis, self.coefficients)

        if self.use_bias:
            output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "max_order": self.max_order,
        })
        return config
