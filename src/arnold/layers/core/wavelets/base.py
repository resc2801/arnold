# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Base class for wavelet-based KAN layers.

This module provides the abstract base class :class:`WaveletBase` for all
wavelet KAN layers, implementing the common scale/translation framework.
"""
from abc import abstractmethod

import tensorflow as tf

from arnold.layers.core.kan_base import KANBase
from arnold.utils.compilation import kan_function
from arnold.utils.constants import PARAM_EPS


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="WaveletBase")
class WaveletBase(KANBase):
    r"""
    Abstract base class for Kolmogorov-Arnold Network layers using wavelets.

    This layer computes:

    .. math::

        y_j = \sum_{i=1}^{d_{\text{in}}} w_{i,j} \cdot \frac{1}{\sqrt{s_j}} \,
              \psi\!\left(\frac{x_i - t_j}{s_j}\right) + b_j

    where :math:`\psi` is the mother wavelet, :math:`s_j > 0` is the learnable scale,
    :math:`t_j` is the learnable translation, and :math:`w_{i,j}` are learnable weights.
    """

    def __init__(
        self,
        units: int,
        scale_init: float | None = None,
        scale_trainable: bool = True,
        translation_init: float | None = None,
        translation_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        super().__init__(units=units, input_clip=input_clip, **kwargs)

        if (scale_init is not None) and (scale_init <= 0):
            raise ValueError("Non-zero, positive value for the initial wavelet scale parameter required!")

        self.scale_init = scale_init
        self.translation_init = translation_init
        self.scale_trainable = scale_trainable
        self.translation_trainable = translation_trainable
        self.scale = None
        self.translation = None
        self.wavelet_weights = None

    def build(self, input_shape):
        super().build(input_shape)

        scale_init = (
            tfk.initializers.Constant(value=tf.math.log(self.scale_init))
            if self.scale_init
            else tfk.initializers.HeNormal()
        )
        translation_init = (
            tfk.initializers.Constant(value=self.translation_init)
            if self.translation_init is not None
            else tfk.initializers.HeNormal()
        )

        self.scale = self.add_weight(
            shape=(1, self.output_dim, self.input_dim),
            initializer=scale_init,
            name="scale_logits",
            trainable=self.scale_trainable,
        )

        self.translation = self.add_weight(
            shape=(1, self.output_dim, self.input_dim),
            initializer=translation_init,
            name="translation",
            trainable=self.translation_trainable,
        )

        self.wavelet_weights = self.add_weight(
            shape=(self.output_dim, self.input_dim),
            initializer=tfk.initializers.HeUniform(),
            regularizer=self.kernel_regularizer,
            name="wavelet_weights",
            trainable=True,
        )

    def call(self, inputs):
        x = self._preprocess_inputs(inputs)
        original_dtype = x.dtype
        leading_shape = tf.shape(x)[:-1]

        compute_dtype = self.effective_compute_dtype
        if x.dtype != compute_dtype:
            x = tf.cast(x, compute_dtype)

        scale = tf.nn.softplus(self.scale) + tf.cast(PARAM_EPS, compute_dtype)
        x_scaled = tf.math.divide(
            tf.expand_dims(x, axis=1) - self.translation,
            scale,
        )

        daughter_wavelets = self.mother_wavelet(x_scaled) / tf.math.sqrt(scale)

        y_flat = tf.einsum(
            "boi,oi->bo",
            daughter_wavelets,
            self.wavelet_weights,
            optimize="auto",
        )

        if y_flat.dtype != original_dtype:
            y_flat = tf.cast(y_flat, original_dtype)

        y_flat = self._apply_activation_and_bias(y_flat)
        return tf.reshape(y_flat, tf.concat([leading_shape, [self.output_dim]], axis=0))

    @abstractmethod
    def mother_wavelet(self, x):
        r"""Compute the mother wavelet for input tensor `x`."""
        raise NotImplementedError(
            f"Layer {self.__class__.__name__} does not have a `mother_wavelet()` method implemented."
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "scale_init": self.scale_init,
                "translation_init": self.translation_init,
                "scale_trainable": self.scale_trainable,
                "translation_trainable": self.translation_trainable,
            }
        )
        return config
