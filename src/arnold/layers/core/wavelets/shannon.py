# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""Shannon (sinc) wavelet KAN layer."""
import numpy as np
import tensorflow as tf

from arnold.layers.core.wavelets.base import WaveletBase
from arnold.utils.compilation import kan_function


kan_fn = kan_function()


class Shannon(WaveletBase):
    r"""KAN layer using Shannon wavelets with Hamming window."""

    def __init__(self, *args, units: int, input_clip=None, **kwargs):
        super().__init__(*args, units=units, input_clip=input_clip, **kwargs)
        self._hamming_window = None
        self._normalization = None

    def build(self, input_shape):
        super().build(input_shape)
        dtype = tf.dtypes.as_dtype(self.compute_dtype)
        self._hamming_window = tf.signal.hamming_window(self.input_dim, periodic=False, dtype=dtype)
        self._normalization = tf.sqrt(tf.reduce_sum(tf.square(self._hamming_window)))

    @kan_fn
    def mother_wavelet(self, x):
        pi_x = np.pi * x
        sinc_x = tf.where(tf.abs(x) < 1e-8, tf.ones_like(x), tf.math.sin(pi_x) / pi_x)
        window = tf.cast(self._hamming_window, x.dtype)
        norm = tf.cast(self._normalization, x.dtype)
        return (sinc_x * window) / (norm + 1e-8)

    def get_config(self):
        return super().get_config()
