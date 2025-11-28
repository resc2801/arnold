# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Power RBF Layer
===============

Kolmogorov-Arnold Network layer using the Power radial basis function.
"""

import tensorflow as tf

from arnold.layers.core.rbf.base import RBFBase
from arnold.utils.compilation import kan_function


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="PowerRBF")
class PowerRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the Power radial basis function.

    .. math::

        \phi(r) = r^{p}, \quad p \in \mathbb{R}, r = \lVert x - x_{i} \rVert

    Stability: exponent :math:`p` is constrained to be positive via
    ``softplus``; inputs are floored to avoid ``0**p`` underflow.
    """

    def __init__(
        self,
        *,
        units: int,
        power_init: float | None = None,
        power_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        units : int
            Output dimensionality.
        power_init : float, optional
            Initial exponent :math:`p`; must be positive. Defaults to RandomNormal logits.
        power_trainable : bool
            Whether :math:`p` is trainable.
        input_clip : tuple[float, float], optional
            Optional input clamp.
        **kwargs :
            Forwarded to :class:`RBFBase`.
        """
        super().__init__(units=units, input_clip=input_clip, **kwargs)

        self.power_init = power_init
        self.power_trainable = power_trainable
        self.power_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        initializer = (
            tfk.initializers.Constant(value=tf.math.log(self.power_init))
            if self.power_init
            else tfk.initializers.RandomNormal()
        )
        self.power_logits = self.add_weight(
            shape=(1, self.input_dim, 1),
            initializer=initializer,
            trainable=self.power_trainable,
            name="power_logits",
        )

    @kan_fn
    def get_kernels(self, r):
        power = self._positive_from_logits(self.power_logits)
        r_safe = tf.maximum(r, tf.cast(1e-6, r.dtype))
        return tf.math.pow(r_safe, power)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "power_init": self.power_init,
                "power_trainable": self.power_trainable,
            }
        )
        return config
