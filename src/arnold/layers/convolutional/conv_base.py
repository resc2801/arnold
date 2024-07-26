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
        super().__init__(*args, name='kan_conv_{}'.format(kernel_type), **kwargs)

        
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
        # self.input_spec = tfkl.InputSpec(
        #     min_ndim=self.rank + 2, axes={channel_axis: input_channel}
        # )
        # if input_channel % self.groups != 0:
        #     raise ValueError(
        #         "The number of input channels must be evenly divisible by "
        #         f"the number of groups. Received groups={self.groups}, but the "
        #         f"input has {input_channel} channels (full input shape is "
        #         f"{input_shape})."
        #     )
        # self.kernel_shape = self.kernel_size + (
        #     input_channel // self.groups,
        #     self.filters,
        # )

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




        
        # filters = tf.reshape(self._kernel(inputs), (-1, ) + self.kernel_shape)

        # # Given an input tensor of shape batch_shape + [in_height, in_width, in_channels] and 
        # # a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels], 
        # # this op performs the following:
        # # 1. Flattens the filter to a 2-D matrix with shape [filter_height * filter_width * in_channels, output_channels].
        # # 2. Extracts image patches from the input tensor to form a virtual tensor of shape [batch, out_height, out_width, filter_height * filter_width * in_channels].
        # # 3. For each patch, right-multiplies the filter matrix and the image patch vector.
        # return tf.nn.conv2d(
        #     # A Tensor of rank at least 4. The dimension order is interpreted according to the value of data_format; 
        #     # with the all-but-inner-3 dimensions acting as batch dimensions. See below for details.
        #     inputs,
        #     # A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
        #     filters,
        #     # An int or list of ints that has length 1, 2 or 4. The stride of the sliding window for each dimension of input. 
        #     # If a single value is given it is replicated in the H and W dimension. By default the N and C dimensions are set to 1. 
        #     # The dimension order is determined by the value of data_format
        #     self.strides,
        #     # Either the string "SAME" or "VALID" indicating the type of padding algorithm to use, or 
        #     # a list indicating the explicit paddings at the start and end of each dimension.
        #     "SAME", #self.padding,
        #     # With the default format "NHWC", the data is stored in the order of: batch_shape + [height, width, channels]. 
        #     # Alternatively, the format could be "NCHW", the data storage order of: batch_shape + [channels, height, width].
        #     "NHWC", #self.data_format,
        #     # An int or list of ints that has length 1, 2 or 4, defaults to 1. The dilation factor for each dimension of input. 
        #     # If a single value is given it is replicated in the H and W dimension. By default the N and C dimensions are set to 1. 
        #     # If set to k > 1, there will be k-1 skipped cells between each filter element on that dimension. 
        #     self.dilation_rate,
        #     # A name for the operation (optional).
        #     name=None
        # )
