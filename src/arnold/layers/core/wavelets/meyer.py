# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""Meyer wavelet KAN layer."""
import numpy as np
import tensorflow as tf

from arnold.layers.core.wavelets.base import WaveletBase
from arnold.utils.compilation import kan_function

kan_fn = kan_function()


class Meyer(WaveletBase):
    r"""KAN layer using a Meyer wavelet."""

    def __init__(self, *args, units: int, input_clip=None, **kwargs):
        super().__init__(*args, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def mother_wavelet(self, x):
        def __nu(t):
            return t**4 * (35 - 84 * t + 70 * t**2 - 20 * t**3)

        def __meyer_aux(v):
            return tf.where(
                v <= 1 / 2, tf.ones_like(v), tf.where(v >= 1, tf.zeros_like(v), tf.cos(np.pi / 2 * __nu(2 * v - 1)))
            )

        v = tf.abs(x)
        return tf.sin(np.pi * v) * __meyer_aux(v)
