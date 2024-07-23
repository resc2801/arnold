from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

tfk = tf.keras
tfkl = tfk.layers


class WaveletBase(tfkl.Layer, ABC):
    """
    Abstract base class for Kolmogorov-Arnold Network layer using wavelets.
    """

    def __init__(self, 
                 input_dim, output_dim,
                 *args,
                 tanh_x=True,
                 **kwargs):

        super().__init__(*args, **kwargs)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tanh_x = tanh_x

        # Parameters for wavelet transformation
        self.scale = self.add_weight(
            shape=(self.output_dim, self.input_dim),
            initializer=tfk.initializers.Ones(),
            name='scale',
            trainable=True
        )

        self.translation = self.add_weight(
            shape=(self.output_dim, self.input_dim),
            initializer=tfk.initializers.Zeros(),
            name='translation',
            trainable=True
        )

        # Linear weights for combining outputs
        self.wavelet_weights = self.add_weight(
            shape=(self.output_dim, self.input_dim),
            initializer=tfk.initializers.HeUniform(),
            name='wavelet_weights',
            trainable=True
        )

    
    def call(self, inputs):
        
        # Normalize x to [-1, 1] using tanh
        x = tf.tanh(inputs) if self.tanh_x else inputs

        x = tf.math.divide(
            tf.math.subtract(
                # (batch_size, 1, self.input_dim),
                tf.expand_dims(inputs, axis=1),
                # (batch_size, output_dim, input_dim)
                tf.tile(
                    tf.expand_dims(self.translation, axis=0), 
                    [tf.shape(inputs)[0], 1, 1]
                )   
            ),
            # (batch_size, output_dim, input_dim)
            tf.tile(
                tf.expand_dims(self.scale, axis=0), 
                [tf.shape(inputs)[0], 1, 1]
            )             
        )

        wavelets = self.get_wavelets(x)

        wavelets_weighted = wavelets * tf.tile(
            tf.expand_dims(self.wavelet_weights, axis=0),
            [tf.shape(wavelets)[0], 1, 1]
        )
        
        return tf.reduce_sum(wavelets_weighted, axis=-1)


    @abstractmethod
    def get_wavelets(self, x):
        pass
