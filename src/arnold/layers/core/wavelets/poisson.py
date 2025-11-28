# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""Poisson wavelet KAN layer."""
import numpy as np
import tensorflow as tf

from arnold.layers.core.wavelets.base import WaveletBase
from arnold.utils.compilation import kan_function

kan_fn = kan_function()


class Poisson(WaveletBase):
    r"""KAN layer using Poisson wavelets :math:`\psi(t)=\frac{1}{\pi}\frac{1-t^2}{(1+t^2)^2}`."""

    def __init__(self, *args, units: int, input_clip=None, **kwargs):
        super().__init__(*args, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def mother_wavelet(self, x):
        return (1 / np.pi) * tf.math.divide(
            tf.math.subtract(1.0, tf.square(x)), tf.square(tf.math.add(1.0, tf.square(x)))
        )
