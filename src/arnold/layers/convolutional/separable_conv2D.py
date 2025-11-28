## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
import importlib

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl

from arnold.utils.compilation import kan_function


@tfk.utils.register_keras_serializable(package="arnold", name="SeparableConv2D")
class SeparableConv2D(tfkl.Layer):
    """Depthwise-separable 2D convolution using KAN kernels for pointwise mixing."""

    def __init__(
        self,
        *,
        filters: int,
        kernel_size,
        strides,
        dilation_rate,
        data_format,
        padding,
        groups,
        kernel_type: str,
        kernel_args=None,
        **kwargs,
    ):
        """Initialize the separable convolution layer.

        Parameters
        ----------
        filters : int
            Number of output channels.
        kernel_size : tuple[int, int]
            Spatial kernel size.
        strides : tuple[int, int]
            Stride along height and width.
        dilation_rate : tuple[int, int]
            Dilation along height and width.
        data_format : {"channels_last", "channels_first"}
            Tensor layout.
        padding : {"VALID", "SAME"}
            Padding strategy.
        groups : int
            Number of channel groups (must divide both input and output channels).
        kernel_type : str
            Name of the KAN kernel class exported from ``arnold.layers``.
        kernel_args : dict, optional
            Extra keyword arguments forwarded to the KAN kernel class.
        """
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = tuple(kernel_size)
        self.strides = tuple(strides)
        self.rates = tuple(dilation_rate)
        self.padding = padding
        self.groups = groups
        self.data_format = data_format

        self.kernel_type = kernel_type
        self.kernel_args = kernel_args or {}
        self._kernel = None

    def build(self, input_shape):
        if self.groups < 1:
            raise ValueError("`groups` must be >= 1.")
        if self.filters % self.groups != 0:
            raise ValueError("`filters` must be divisible by `groups`.")
        if self.data_format == "channels_last":
            input_channel = int(input_shape[-1])
        else:
            input_channel = int(input_shape[1])
        if input_channel % self.groups != 0:
            raise ValueError("Input channels must be divisible by `groups`.")

        module = importlib.import_module("arnold.layers")
        self._kernel = getattr(module, self.kernel_type)(
            input_dim=(input_channel // self.groups) * self.kernel_size[0] * self.kernel_size[1],
            units=(self.filters // self.groups),
            **self.kernel_args,
        )
        self.built = True

    @kan_function(jit_compile=False)
    def call(self, inputs):
        """Extract patches and apply the KAN kernel."""
        channels = tf.shape(inputs)[-1]
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=(1,) + self.kernel_size + (1,),
            strides=(1,) + self.strides + (1,),
            padding=self.padding,
            rates=(1,) + self.rates + (1,),
        )

        outputs = self._kernel(
            tf.reshape(
                patches,
                [-1, self.kernel_size[0] * self.kernel_size[1] * channels],
            )
        )
        return tf.reshape(
            outputs,
            [-1, tf.shape(patches)[1], tf.shape(patches)[2], self.filters],
        )
