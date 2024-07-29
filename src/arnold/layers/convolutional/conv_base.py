from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

tfk = tf.keras
tfkl = tfk.layers

import importlib


@tfk.utils.register_keras_serializable(package="arnold", name="ConvBase")
class ConvBase(tfkl.Layer):
    """
    Abstract base class for Kolmogorov-Arnold convolutional layers.
    """

    def __init__(self, 
                 *args, 
                 filters,
                 kernel_size,
                 strides,
                 dilation_rate,
                 data_format,
                 padding,
                 groups,
                 kernel_type:str,
                 kernel_args = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides 
        self.rates = dilation_rate 
        self.padding = padding
        self.groups = groups
        self.data_format = data_format

        self.kernel_type = kernel_type
        self.kernel_args = kernel_args
        self._kernel = None



    def build(self, input_shape):
        if self.data_format == "channels_last":
            channel_axis = -1
            input_channel = input_shape[-1]
        else:
            channel_axis = 1
            input_channel = input_shape[1]

        module = importlib.import_module('arnold.layers')

        self._kernel = getattr(module, self.kernel_type)(
            input_dim=(input_channel // self.groups) * self.kernel_size[0] * self.kernel_size[1],
            output_dim=(self.filters // self.groups)
        ) if self.kernel_args is None else getattr(module, self.kernel_type)(
            input_dim=(input_channel // self.groups) * self.kernel_size[0] * self.kernel_size[1],
            output_dim=(self.filters // self.groups),
            **self.kernel_args
        )

        self.built = True


    def call(self, inputs):

        channels = tf.shape(inputs)[-1]

        # (-1), num_patches_w, num_pachtes_H, kernel_size[0] * kernel_size[1] * channels)
        patches = tf.image.extract_patches(
            images=inputs, 
            sizes=(1,) + self.kernel_size + (1,),
            strides=(1,) + self.strides + (1,),
            padding=self.padding,
            rates=(1,) + self.rates + (1,)
        )

        return tf.reshape(
            self._kernel(
                tf.reshape(
                    patches, 
                    [-1, self.kernel_size[0] * self.kernel_size[1] * channels]
                )
            ),
            [-1, tf.shape(patches)[1], tf.shape(patches)[2], self.filters]
        )
