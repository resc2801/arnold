# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Linear RBF Layer
================

Kolmogorov-Arnold Network layer using the Linear radial basis function.
"""

import tensorflow as tf

from arnold.layers.core.rbf.base import RBFBase


tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold", name="LinearRBF")
class LinearRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the Linear radial basis function.

    .. math::

        \phi(r) = r, \quad r = \lVert x - x_{i} \rVert
    """

    def __init__(self, *, units: int, input_clip=None, **kwargs):
        super().__init__(units=units, input_clip=input_clip, **kwargs)

    @tf.function(
        autograph=True,
        jit_compile=True,
        reduce_retracing=True,
        experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
    )
    def get_kernels(self, r):
        return r
