# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""Morelet (Morlet) wavelet KAN layer."""
import tensorflow as tf

from arnold.layers.core.wavelets.base import WaveletBase
from arnold.utils.compilation import kan_function


tfk = tf.keras
kan_fn = kan_function()


class Morelet(WaveletBase):
    r"""KAN layer using Morelet (Morlet) wavelets."""

    def __init__(
        self,
        *,
        units: int,
        omega_init: float = 5.0,
        omega_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        super().__init__(units=units, input_clip=input_clip, **kwargs)
        self.omega_init = omega_init
        self.omega_trainable = omega_trainable
        self.omega0 = None

    def build(self, input_shape):
        super().build(input_shape)
        self.omega0 = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.omega_init),
            name="central_frequency",
            trainable=self.omega_trainable,
        )

    @kan_fn
    def mother_wavelet(self, x):
        real = tf.cos(self.omega0 * x)
        envelope = tf.exp(-0.5 * x**2)
        return envelope * real

    def get_config(self):
        config = super().get_config()
        config.update({"omega": self.omega_init, "omega_trainable": self.omega_trainable})
        return config
