# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Thin Plate Spline RBF Layer
===========================

Kolmogorov-Arnold Network layer using the Thin Plate Spline radial basis function.
"""

import tensorflow as tf

from arnold.layers.core.rbf.base import RBFBase


tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold", name="ThinPlateSplineRBF")
class ThinPlateSplineRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the Thin plate spline radial basis function.

    .. math::

        \phi(r) = r^{2} \ln(r), \quad r = \lVert x - x_{i} \rVert

    Stability: applies a small floor to :math:`r` before the logarithm to avoid
    ``log(0)`` while preserving gradients.
    """

    def __init__(self, *, units: int, input_clip=None, **kwargs):
        """
        Parameters
        ----------
        units : int
            Output dimensionality.
        input_clip : tuple[float, float] | None
            Optional input clamp.
        """
        super().__init__(units=units, input_clip=input_clip, **kwargs)

    @tf.function(
        autograph=True,
        jit_compile=True,
        reduce_retracing=True,
        experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
    )
    def get_kernels(self, r):
        r"""
        :math:`\phi(r) = r^{2} \ln(r)` with a small floor to avoid :math:`\log(0)`.
        """
        r_safe = tf.maximum(r, tf.cast(1e-6, r.dtype))
        return tf.square(r_safe) * tf.math.log(r_safe)
