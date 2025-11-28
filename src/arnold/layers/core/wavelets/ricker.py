# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""Ricker (Mexican hat) wavelet KAN layer."""
import numpy as np
import tensorflow as tf

from arnold.layers.core.wavelets.base import WaveletBase
from arnold.utils.compilation import kan_function

tfk = tf.keras
kan_fn = kan_function()


class Ricker(WaveletBase):
    r"""KAN layer using the Ricker (Mexican hat) wavelet."""

    def __init__(
        self,
        *,
        units: int,
        sigma_init: float = 1.0,
        sigma_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        super().__init__(units=units, input_clip=input_clip, **kwargs)
        self.sigma_init = sigma_init
        self.sigma_trainable = sigma_trainable
        self.sigma = None

    def build(self, input_shape):
        super().build(input_shape)
        self.sigma = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.sigma_init),
            name="standard_deviation",
            trainable=self.sigma_trainable,
        )

    @kan_fn
    def mother_wavelet(self, x):
        sigma = tf.nn.softplus(self.sigma) + tf.cast(1e-6, x.dtype)
        term1 = 1.0 - tf.square(x / sigma)
        term2 = tf.exp(-0.5 * tf.square(x / sigma))
        return (2 / (tf.math.sqrt(3.0) * np.pi**0.25)) * term1 * term2

    def get_config(self):
        config = super().get_config()
        config.update({"sigma": self.sigma_init, "sigma_trainable": self.sigma_trainable})
        return config
