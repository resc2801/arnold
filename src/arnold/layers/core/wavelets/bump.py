# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""Bump wavelet KAN layer."""
import tensorflow as tf

from arnold.layers.core.wavelets.base import WaveletBase
from arnold.utils.compilation import kan_function

kan_fn = kan_function()


class Bump(WaveletBase):
    r"""KAN layer using Bump wavelets :math:`\psi(x) = \mathbf{I}_{[-1,1]}(x)\, e^{(1-\frac{1}{1-x^2})}`."""

    def __init__(self, *args, units: int, input_clip=None, **kwargs):
        super().__init__(*args, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def mother_wavelet(self, x):
        eps = 1e-07
        x = tf.clip_by_value(x, -1.0 + eps, 1.0 - eps)
        return tf.exp(-1.0 / (1 - x**2))
