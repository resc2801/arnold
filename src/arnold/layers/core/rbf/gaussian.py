# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Gaussian RBF Layer
==================

Kolmogorov-Arnold Network layer using the Gaussian radial basis function.
"""

import tensorflow as tf

from arnold.layers.core.rbf.base import RBFBase


tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold", name="GaussianRBF")
class GaussianRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the Gaussian radial basis function.

    The Gaussian RBF kernel is defined as:

    .. math::

        \phi(r) = \exp\!\left(-(\varepsilon r)^2\right), \quad r = \|x - \mu_k\|

    where :math:`\varepsilon > 0` is the shape parameter controlling the kernel width,
    and :math:`\mu_k` are the grid centers.

    The shape parameter :math:`\varepsilon` is stored in logits and transformed via
    ``softplus`` to ensure positivity and smooth gradient flow.

    Notes
    -----
    - Small :math:`\varepsilon`: wide, smooth kernels → global influence
    - Large :math:`\varepsilon`: narrow, peaked kernels → local influence

    For optimal interpolation, choose ``num_grids`` to roughly match the expected
    number of "features" in your input domain. The kernel width adapts during training.

    See Also
    --------
    MultiquadricRBF : Grows unboundedly, good for global approximation
    InverseMultiquadricRBF : Decays like Gaussian but with polynomial tails
    """

    def __init__(
        self,
        *,
        units: int,
        epsilon_init: float | None = None,
        epsilon_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        units : int
            Output dimensionality.
        epsilon_init : float, optional
            Initial positive shape parameter :math:`\epsilon`.
        epsilon_trainable : bool
            Whether :math:`\epsilon` is trainable.
        input_clip : tuple[float, float], optional
            Optional input clamp.
        **kwargs :
            Forwarded to :class:`RBFBase`.
        """
        super().__init__(units=units, input_clip=input_clip, **kwargs)

        self.epsilon_init = epsilon_init
        self.epsilon_trainable = epsilon_trainable
        self.epsilon_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        initializer = (
            tfk.initializers.Constant(value=tf.math.log(self.epsilon_init))
            if self.epsilon_init
            else tfk.initializers.RandomNormal()
        )
        self.epsilon_logits = self.add_weight(
            shape=(1, self.input_dim, 1),
            initializer=initializer,
            trainable=self.epsilon_trainable,
            name="epsilon_logits",
        )

    @tf.function(
        autograph=True,
        jit_compile=True,
        reduce_retracing=True,
        experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
    )
    def get_kernels(self, r):
        eps_param = self._positive_from_logits(self.epsilon_logits)
        return tf.exp(-((eps_param * r) ** 2))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "epsilon_init": self.epsilon_init,
                "epsilon_trainable": self.epsilon_trainable,
            }
        )
        return config
