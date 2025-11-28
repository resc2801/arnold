# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""Catmull-Rom spline KAN layer."""
import tensorflow as tf

from arnold.layers.core.splines.base import SplineBase
from arnold.utils.compilation import kan_function


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="CatmullRom")
class CatmullRom(SplineBase):
    r"""KAN layer using Catmull-Rom spline basis."""

    def __init__(self, *, units: int, num_knots: int = 8, knot_range: tuple[float, float] = (-1.0, 1.0),
                 trainable_knots: bool = False, input_clip: tuple[float, float] | None = None, **kwargs):
        super().__init__(units=units, num_knots=num_knots, knot_range=knot_range,
                         trainable_knots=trainable_knots, input_clip=input_clip, **kwargs)

    def _get_num_basis_functions(self) -> int:
        return self.num_knots

    @kan_fn
    def spline_basis(self, x: tf.Tensor) -> tf.Tensor:
        knots = tf.cast(self._knots, x.dtype)
        M = tf.constant([[-0.5, 1.5, -1.5, 0.5], [1.0, -2.5, 2.0, -0.5], [-0.5, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=x.dtype)
        knot_min, knot_max = knots[0], knots[-1]
        t_normalized = (x - knot_min) / (knot_max - knot_min + 1e-8) * (self.num_knots - 1)
        t_normalized = tf.clip_by_value(t_normalized, 0.0, self.num_knots - 1.0 - 1e-6)
        segment = tf.floor(t_normalized)
        u = t_normalized - segment
        u2, u3 = u * u, u * u * u
        powers = tf.stack([u3, u2, u, tf.ones_like(u)], axis=-1)
        weights = tf.einsum("...p,pq->...q", powers, M)
        k_indices = tf.range(self.num_knots, dtype=x.dtype)
        segment_exp = tf.expand_dims(segment, axis=-1)
        dist_to_cp = tf.abs(segment_exp - k_indices + 1.0)
        active_mask = tf.cast(dist_to_cp < 2.0, x.dtype)
        rel_pos = k_indices - segment_exp + 1.0
        w0 = weights[..., 0:1] * tf.cast(tf.abs(rel_pos - 0.0) < 0.5, x.dtype)
        w1 = weights[..., 1:2] * tf.cast(tf.abs(rel_pos - 1.0) < 0.5, x.dtype)
        w2 = weights[..., 2:3] * tf.cast(tf.abs(rel_pos - 2.0) < 0.5, x.dtype)
        w3 = weights[..., 3:4] * tf.cast(tf.abs(rel_pos - 3.0) < 0.5, x.dtype)
        return (w0 + w1 + w2 + w3) * active_mask

    def get_config(self):
        return super().get_config()
