# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""Haar wavelet KAN layer."""
import tensorflow as tf

from arnold.layers.core.wavelets.base import WaveletBase
from arnold.utils.compilation import kan_function


tfk = tf.keras
kan_fn = kan_function()


class Haar(WaveletBase):
    r"""KAN layer using the Haar wavelet (smooth sigmoid approximation)."""

    def __init__(
        self,
        *,
        units: int,
        sharpness: float = 20.0,
        sharpness_trainable: bool = False,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        super().__init__(units=units, input_clip=input_clip, **kwargs)
        self.sharpness_init = sharpness
        self.sharpness_trainable = sharpness_trainable
        self._sharpness = None

    def build(self, input_shape):
        super().build(input_shape)
        self._sharpness = self.add_weight(
            shape=(),
            initializer=tfk.initializers.Constant(self.sharpness_init),
            name="sharpness",
            trainable=self.sharpness_trainable,
        )

    @kan_fn
    def mother_wavelet(self, x):
        k = tf.nn.softplus(self._sharpness) + 1.0
        step_0 = tf.sigmoid(k * x)
        step_half = tf.sigmoid(k * (x - 0.5))
        step_1 = tf.sigmoid(k * (x - 1.0))
        return step_0 - 2.0 * step_half + step_1

    def get_config(self):
        config = super().get_config()
        config.update({"sharpness": self.sharpness_init, "sharpness_trainable": self.sharpness_trainable})
        return config
