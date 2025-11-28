# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Base class for geometric KAN layers.

This module provides the abstract base class :class:`GeometricBase` for all
geometric KAN layers using basis functions on geometric domains.
"""
from abc import abstractmethod

import tensorflow as tf

from arnold.layers.core.kan_base import KANBase


tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold", name="GeometricBase")
class GeometricBase(KANBase):
    r"""
    Abstract base class for Kolmogorov-Arnold Network layers using geometric bases.

    Geometric bases are defined on geometric domains (disk, sphere, hypersphere)
    and are typically eigenfunctions of Laplace-Beltrami operators.

    This layer computes:

    .. math::

        y_j = \sum_{i=1}^{d_{\text{in}}} \sum_{k=0}^{n_{\text{basis}}-1}
              c_{i,j,k} \, \phi_k(x_i) + b_j

    where :math:`\phi_k` are geometric basis functions (Zernike, spherical harmonics, etc.)
    and :math:`c_{i,j,k}` are learnable coefficients.

    Parameters
    ----------
    units : int
        Output dimensionality.
    max_degree : int
        Maximum degree/order of the geometric basis expansion.
    input_clip : tuple[float, float] | None
        Optional input clamp before basis evaluation.

    Subclasses
    ----------
    Subclasses must implement:
    - :meth:`_get_num_basis_functions`: Return count of basis functions
    - :meth:`geometric_basis`: Compute basis functions at input locations
    """

    def __init__(
        self,
        units: int,
        max_degree: int = 8,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        super().__init__(units=units, input_clip=input_clip, **kwargs)

        if max_degree < 0:
            raise ValueError(f"max_degree must be >= 0, got {max_degree}")

        self.max_degree = max_degree
        self._geometric_coeffs = None

    def build(self, input_shape):
        super().build(input_shape)

        num_basis = self._get_num_basis_functions()
        self._geometric_coeffs = self.add_weight(
            shape=(self.output_dim, self.input_dim, num_basis),
            initializer=tfk.initializers.GlorotUniform(),
            regularizer=self.kernel_regularizer,
            name="geometric_coeffs",
            trainable=True,
        )

    @abstractmethod
    def _get_num_basis_functions(self) -> int:
        """Return the number of basis functions for this geometric type."""
        raise NotImplementedError

    @abstractmethod
    def geometric_basis(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute geometric basis functions at input locations.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (..., input_dim).

        Returns
        -------
        tf.Tensor
            Basis values of shape (..., input_dim, num_basis).
        """
        raise NotImplementedError

    def call(self, inputs):
        x = self._preprocess_inputs(inputs)
        original_dtype = x.dtype
        leading_shape = tf.shape(x)[:-1]
        batch_size = tf.reduce_prod(leading_shape)

        compute_dtype = self.effective_compute_dtype
        if x.dtype != compute_dtype:
            x = tf.cast(x, compute_dtype)

        # Flatten leading dimensions: (..., input_dim) -> (batch, input_dim)
        x_flat = tf.reshape(x, [batch_size, self.input_dim])

        # Evaluate geometric basis: (batch, input_dim, num_basis)
        basis = self.geometric_basis(x_flat)

        # Contract: coeffs[o, i, k] * basis[b, i, k] -> output[b, o]
        y = tf.einsum(
            "oik,bik->bo",
            tf.cast(self._geometric_coeffs, compute_dtype),
            basis,
            optimize="auto",
        )

        if y.dtype != original_dtype:
            y = tf.cast(y, original_dtype)

        y = self._apply_activation_and_bias(y)
        return tf.reshape(y, tf.concat([leading_shape, [self.output_dim]], axis=0))

    def get_config(self):
        config = super().get_config()
        config.update({"max_degree": self.max_degree})
        return config
