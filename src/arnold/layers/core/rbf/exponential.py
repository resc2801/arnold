# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Exponential RBF Layer
=====================

Kolmogorov-Arnold Network layer using the Exponential radial basis function.
"""

import tensorflow as tf

from arnold.layers.core.rbf.base import RBFBase
from arnold.utils.compilation import kan_function


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="ExponentialRBF")
class ExponentialRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the Exponential radial basis function.

    .. math::

        \phi(r) = e^{ - \frac{r}{\sigma}}, \quad \sigma \in \mathbb{R}, r = \lVert x - x_{i} \rVert

    Stability: :math:`\sigma` is stored as logits and mapped to
    ``softplus(sigma_logits) + eps`` to keep it positive and well-conditioned.
    """

    def __init__(
        self,
        *,
        units: int,
        sigma_init: float | None = None,
        sigma_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        units : int
            Output dimensionality.
        sigma_init : float | None
            Initial value for shape parameter :math:`\sigma`; defaults to RandomNormal when None.
        sigma_trainable : bool
            Whether :math:`\sigma` is trainable.
        input_clip : tuple[float, float] | None
            Optional input clamp.
        **kwargs :
            Forwarded to :class:`RBFBase`.
        """
        super().__init__(units=units, input_clip=input_clip, **kwargs)
        self.sigma_init = sigma_init
        self.sigma_trainable = sigma_trainable
        self.sigma_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        initializer = (
            tfk.initializers.Constant(value=tf.math.log(self.sigma_init))
            if self.sigma_init
            else tfk.initializers.RandomNormal()
        )
        self.sigma_logits = self.add_weight(
            shape=(1, self.input_dim, 1),
            initializer=initializer,
            trainable=self.sigma_trainable,
            name="sigma_logits",
        )

    @kan_fn
    def get_kernels(self, r):
        sigma = self._positive_from_logits(self.sigma_logits)
        return tf.math.exp(-(r / sigma))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sigma_init": self.sigma_init,
                "sigma_trainable": self.sigma_trainable,
            }
        )
        return config
