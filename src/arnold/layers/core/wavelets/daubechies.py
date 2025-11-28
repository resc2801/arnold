# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""Daubechies wavelet KAN layer."""
from typing import Literal

import numpy as np
import tensorflow as tf

from arnold.layers.core.wavelets.base import WaveletBase
from arnold.layers.core.wavelets.coefficients import DAUBECHIES_COEFFICIENTS
from arnold.utils.compilation import kan_function

kan_fn = kan_function()


class Daubechies(WaveletBase):
    r"""KAN layer using Daubechies wavelets (Gaussian-mixture approximation)."""

    def __init__(
        self,
        *,
        units: int,
        order: Literal[1, 2, 3, 4, 5, 6, 7, 8, 10] = 4,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        if order not in DAUBECHIES_COEFFICIENTS:
            available = sorted(DAUBECHIES_COEFFICIENTS.keys())
            raise ValueError(f"Daubechies order must be in {available}, got {order}")
        super().__init__(units=units, input_clip=input_clip, **kwargs)
        self.order = order
        self._g = None
        self._centers = None

    def build(self, input_shape):
        super().build(input_shape)
        h = DAUBECHIES_COEFFICIENTS[self.order].astype(np.float32)
        g = np.array([(-1) ** n * h[len(h) - 1 - n] for n in range(len(h))])
        self._g = tf.constant(g, dtype=self.compute_dtype, name="wavelet_filter")
        self._centers = tf.constant(np.arange(len(h), dtype=np.float32), dtype=self.compute_dtype, name="centers")
        self._support = float(len(h) - 1)

    @kan_fn
    def mother_wavelet(self, x):
        g = tf.cast(self._g, x.dtype)
        centers = tf.cast(self._centers, x.dtype)
        support = tf.cast(self._support, x.dtype)
        t = x * support
        t_expanded = tf.expand_dims(t, axis=-1)
        shifts = t_expanded - centers
        sigma = 0.5 + 0.1 * support
        gaussians = tf.exp(-0.5 * tf.square(shifts / sigma))
        weighted = gaussians * g
        psi = tf.reduce_sum(weighted, axis=-1)
        return psi / (tf.sqrt(support) + 1e-8)

    def get_config(self):
        config = super().get_config()
        config.update({"order": self.order})
        return config
