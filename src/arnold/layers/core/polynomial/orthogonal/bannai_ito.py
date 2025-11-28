# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Bannai-Ito polynomials.

This module implements Bannai-Ito polynomials :math:`B_n(x; \rho_1, \rho_2, r_1, r_2)`,
which are orthogonal polynomials of a discrete argument from the Bannai-Ito scheme.

Mathematical Background
-----------------------
The Bannai-Ito polynomials are a family of orthogonal polynomials with respect to
a discrete measure. They have a three-term recurrence relation with coefficients
depending on four parameters :math:`\rho_1, \rho_2, r_1, r_2`.

Note
----
All parameters :math:`\rho_1, \rho_2, r_1, r_2` remain unconstrained.

References
----------
.. [DLMF] NIST Digital Library of Mathematical Functions, §18.36
   https://dlmf.nist.gov/18.36
"""

import tensorflow as tf

from arnold.utils.compilation import kan_function
from arnold.utils.weights import create_trainable_param

from ..poly_base import PolynomialBase


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="BannaiIto")
class BannaiIto(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using the Bannai-Ito polynomials.

    All parameters :math:`\rho_1, \rho_2, r_1, r_2` remain unconstrained.

    Attributes
    ----------
    rho1 : tf.Variable
        Parameter :math:`\rho_1`.
    rho2 : tf.Variable
        Parameter :math:`\rho_2`.
    r1 : tf.Variable
        Parameter :math:`r_1`.
    r2 : tf.Variable
        Parameter :math:`r_2`.
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        rho1_init: float | None = None,
        rho1_trainable: bool = True,
        rho2_init: float | None = None,
        rho2_trainable: bool = True,
        r1_init: float | None = None,
        r1_trainable: bool = True,
        r2_init: float | None = None,
        r2_trainable: bool = True,
        input_clip=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        degree : int
            Maximum polynomial degree.
        units : int
            Output dimensionality.
        rho1_init, rho2_init, r1_init, r2_init : float | None
            Initial values for parameters; default to RandomNormal when None.
        rho1_trainable, rho2_trainable, r1_trainable, r2_trainable : bool
            Trainability flags.
        input_clip : tuple[float, float] | None
            Optional input clamp.
        orthonormal : bool, default False
            Placeholder; not implemented for Bannai-Ito yet.
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

        self.rho1_init = rho1_init
        self.rho1_trainable = rho1_trainable
        self.rho2_init = rho2_init
        self.rho2_trainable = rho2_trainable
        self.r1_init = r1_init
        self.r1_trainable = r1_trainable
        self.r2_init = r2_init
        self.r2_trainable = r2_trainable
        self.orthonormal = kwargs.pop("orthonormal", False)

        self.rho1 = None
        self.rho2 = None
        self.r1 = None
        self.r2 = None

    def build(self, input_shape):
        super().build(input_shape)

        self.rho1 = create_trainable_param(self, "rho1", self.rho1_init, trainable=self.rho1_trainable)
        self.rho2 = create_trainable_param(self, "rho2", self.rho2_init, trainable=self.rho2_trainable)
        self.r1 = create_trainable_param(self, "r1", self.r1_init, trainable=self.r1_trainable)
        self.r2 = create_trainable_param(self, "r2", self.r2_init, trainable=self.r2_trainable)

    @kan_fn
    def pseudo_vandermonde(self, x):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)

        if self.orthonormal:
            raise NotImplementedError("Orthonormal scaling for Bannai-Ito is not yet implemented.")

        rho1 = tf.cast(self.rho1, tf.float64)
        rho2 = tf.cast(self.rho2, tf.float64)
        r1 = tf.cast(self.r1, tf.float64)
        r2 = tf.cast(self.r2, tf.float64)

        bannai_ito_basis = [tf.ones_like(x, dtype=tf.float64)]

        if self.degree > 0:
            bannai_ito_basis.append(x - rho1)

        for n in range(2, self.degree + 1):
            An = (n + rho1 + rho2) * (n + r1 + r2) / ((2 * n + rho1 + rho2 + r1 + r2 - 1) * (2 * n + rho1 + rho2 + r1 + r2))
            Cn = n * (n + rho1 + rho2 - 1) / ((2 * n + rho1 + rho2 + r1 + r2 - 2) * (2 * n + rho1 + rho2 + r1 + r2 - 1))
            bannai_ito_basis.append(
                (x - An) * bannai_ito_basis[n - 1] - Cn * bannai_ito_basis[n - 2]
            )

        return tf.cast(tf.stack(bannai_ito_basis, axis=-1), orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rho1_init": self.rho1_init,
                "rho1_trainable": self.rho1_trainable,
                "rho2_init": self.rho2_init,
                "rho2_trainable": self.rho2_trainable,
                "r1_init": self.r1_init,
                "r1_trainable": self.r1_trainable,
                "r2_init": self.r2_init,
                "r2_trainable": self.r2_trainable,
            }
        )
        return config


__all__ = [
    "BannaiIto",
]
