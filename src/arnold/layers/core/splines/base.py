# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Base class for spline-based KAN layers.
"""
from abc import abstractmethod

import numpy as np
import tensorflow as tf

from arnold.layers.core.kan_base import KANBase

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold", name="SplineBase")
class SplineBase(KANBase):
    r"""Abstract base class for KAN layers using spline bases."""

    def __init__(
        self,
        units: int,
        num_knots: int = 8,
        knot_range: tuple[float, float] = (-1.0, 1.0),
        trainable_knots: bool = False,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        super().__init__(units=units, input_clip=input_clip, **kwargs)
        if num_knots < 2:
            raise ValueError(f"num_knots must be >= 2, got {num_knots}")
        self.num_knots = num_knots
        self.knot_range = knot_range
        self.trainable_knots = trainable_knots
        self._knots = None
        self._spline_coeffs = None

    def build(self, input_shape):
        super().build(input_shape)
        knot_init = np.linspace(self.knot_range[0], self.knot_range[1], self.num_knots).astype(np.float32)
        self._knots = self.add_weight(
            shape=(self.num_knots,),
            initializer=tfk.initializers.Constant(knot_init),
            name="knots",
            trainable=self.trainable_knots,
        )
        num_basis = self._get_num_basis_functions()
        self._spline_coeffs = self.add_weight(
            shape=(self.output_dim, self.input_dim, num_basis),
            initializer=tfk.initializers.HeUniform(),
            regularizer=self.kernel_regularizer,
            name="spline_coeffs",
            trainable=True,
        )

    @abstractmethod
    def _get_num_basis_functions(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def spline_basis(self, x: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def call(self, inputs):
        x = self._preprocess_inputs(inputs)
        original_dtype = x.dtype
        leading_shape = tf.shape(x)[:-1]
        compute_dtype = self.effective_compute_dtype
        if x.dtype != compute_dtype:
            x = tf.cast(x, compute_dtype)
        basis = self.spline_basis(x)
        y = tf.einsum("oik,bik->bo", tf.cast(self._spline_coeffs, compute_dtype), basis, optimize="auto")
        if y.dtype != original_dtype:
            y = tf.cast(y, original_dtype)
        y = self._apply_activation_and_bias(y)
        return tf.reshape(y, tf.concat([leading_shape, [self.output_dim]], axis=0))

    def get_config(self):
        config = super().get_config()
        config.update({"num_knots": self.num_knots, "knot_range": self.knot_range, "trainable_knots": self.trainable_knots})
        return config
