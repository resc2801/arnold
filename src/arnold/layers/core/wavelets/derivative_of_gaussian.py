# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""Derivative of Gaussian wavelet KAN layer."""
import tensorflow as tf

from arnold.layers.core.wavelets.base import WaveletBase
from arnold.utils.compilation import kan_function

kan_fn = kan_function()


class DerivativeOfGaussian(WaveletBase):
    r"""KAN layer using the first derivative of a Gaussian wavelet."""

    def __init__(self, *args, units: int, input_clip=None, **kwargs):
        super().__init__(*args, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def mother_wavelet(self, x):
        return -x * tf.exp(-0.5 * x**2)
