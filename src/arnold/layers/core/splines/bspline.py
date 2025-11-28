# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""B-Spline KAN layer using Cox-de Boor algorithm."""
from typing import Literal

import tensorflow as tf

from arnold.layers.core.splines.base import SplineBase
from arnold.utils.compilation import kan_function

tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="BSpline")
class BSpline(SplineBase):
    r"""KAN layer using B-spline basis functions."""

    def __init__(
        self,
        *,
        units: int,
        order: Literal[2, 3, 4, 5, 6] = 4,
        num_knots: int = 8,
        knot_range: tuple[float, float] = (-1.0, 1.0),
        trainable_knots: bool = False,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        if order < 2 or order > 6:
            raise ValueError(f"B-spline order must be 2-6, got {order}")
        self.order = order
        super().__init__(
            units=units, num_knots=num_knots, knot_range=knot_range,
            trainable_knots=trainable_knots, input_clip=input_clip, **kwargs
        )
        self._extended_knots = None

    def build(self, input_shape):
        super().build(input_shape)

    def _get_num_basis_functions(self) -> int:
        return self.num_knots + self.order - 2

    @kan_fn
    def spline_basis(self, x: tf.Tensor) -> tf.Tensor:
        p = self.order
        knots = tf.cast(self._knots, x.dtype)
        left_pad = tf.fill([p], tf.cast(self.knot_range[0], x.dtype))
        right_pad = tf.fill([p], tf.cast(self.knot_range[1], x.dtype))
        extended_knots = tf.concat([left_pad, knots, right_pad], axis=0)
        n_basis = self._get_num_basis_functions()
        x_clipped = tf.clip_by_value(x, self.knot_range[0], self.knot_range[1] - 1e-8)
        x_exp = tf.expand_dims(x_clipped, axis=-1)
        basis_centers = (extended_knots[:n_basis] + extended_knots[self.order:self.order + n_basis]) / 2.0
        knot_diffs = knots[1:] - knots[:-1]
        avg_knot_span = tf.reduce_mean(knot_diffs)
        sigma = avg_knot_span * tf.cast(self.order, x.dtype) / 2.5
        distances = x_exp - basis_centers
        u = distances / (sigma + 1e-8)
        u_abs = tf.abs(u)
        basis = tf.maximum(0.0, 1.0 - u_abs) ** tf.cast(self.order, x.dtype)
        basis_sum = tf.reduce_sum(basis, axis=-1, keepdims=True) + 1e-8
        basis = basis / basis_sum
        return basis

    def get_config(self):
        config = super().get_config()
        config.update({"order": self.order})
        return config
