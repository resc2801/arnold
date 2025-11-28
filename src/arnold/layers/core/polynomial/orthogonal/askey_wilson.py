# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Askey-Wilson polynomials.

This module implements Askey-Wilson polynomials :math:`p_n(\cos\theta; a, b, c, d | q)`,
which are at the top of the q-Askey scheme.

Mathematical Background
-----------------------
The Askey-Wilson polynomials are the most general classical orthogonal polynomials
in the q-analog sense. They satisfy a three-term recurrence with coefficients
depending on q and four parameters a, b, c, d.

Parameters must satisfy :math:`|q| < 1` for stability.

References
----------
.. [DLMF] NIST Digital Library of Mathematical Functions, §18.28
   https://dlmf.nist.gov/18.28
"""

import tensorflow as tf

from arnold.utils.compilation import kan_function
from arnold.utils.constants import PARAM_EPS
from arnold.utils.weights import create_trainable_param

from ..poly_base import PolynomialBase


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="AskeyWilson")
class AskeyWilson(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using the Askey-Wilson polynomials.

    Parameters must satisfy :math:`|q| < 1` for stability; inputs are unbounded.
    Internally clamps ``q`` into ``(-1 + eps, 1 - eps)`` to avoid division by
    zero in the recurrence.
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        a_init: float | None = None,
        a_trainable: bool = True,
        b_init: float | None = None,
        b_trainable: bool = True,
        c_init: float | None = None,
        c_trainable: bool = True,
        d_init: float | None = None,
        d_trainable: bool = True,
        q_init: float | None = None,
        q_trainable: bool = True,
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
        a_init, b_init, c_init, d_init : float | None
            Initial values for parameters; default to RandomNormal when None.
        a_trainable, b_trainable, c_trainable, d_trainable : bool
            Trainability flags.
        q_init : float | None
            Initial value for ``q``; defaults to RandomNormal when None.
        q_trainable : bool
            Trainability flag for ``q``.
        input_clip : tuple[float, float] | None
            Optional input clamp.
        orthonormal : bool, default False
            Placeholder; not implemented for Askey-Wilson yet.
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

        self.a_init = a_init
        self.a_trainable = a_trainable
        self.b_init = b_init
        self.b_trainable = b_trainable
        self.c_init = c_init
        self.c_trainable = c_trainable
        self.d_init = d_init
        self.d_trainable = d_trainable
        self.q_init = q_init
        self.q_trainable = q_trainable
        self.orthonormal = kwargs.pop("orthonormal", False)

        self.a = None
        self.b = None
        self.c = None
        self.d = None
        self.q = None

    def build(self, input_shape):
        super().build(input_shape)

        self.a = create_trainable_param(self, "a", self.a_init, trainable=self.a_trainable)
        self.b = create_trainable_param(self, "b", self.b_init, trainable=self.b_trainable)
        self.c = create_trainable_param(self, "c", self.c_init, trainable=self.c_trainable)
        self.d = create_trainable_param(self, "d", self.d_init, trainable=self.d_trainable)
        self.q = create_trainable_param(self, "q", self.q_init, trainable=self.q_trainable)

    @kan_fn
    def pseudo_vandermonde(self, x):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        eps = tf.constant(PARAM_EPS, dtype=tf.float64)
        q = tf.clip_by_value(tf.cast(self.q, tf.float64), -1.0 + eps, 1.0 - eps)
        # Note: tf.clip_by_value ensures |q| < 1; no runtime assertion needed
        if self.orthonormal:
            raise NotImplementedError("Orthonormal scaling for Askey-Wilson is not yet implemented.")
        a = tf.cast(self.a, tf.float64)
        b = tf.cast(self.b, tf.float64)
        c = tf.cast(self.c, tf.float64)
        d = tf.cast(self.d, tf.float64)
        abcd = a * b * c * d
        ab = a * b
        cd = c * d

        # Precompute q powers up to 2*degree for efficiency
        max_power = 2 * self.degree + 1
        q_powers = tf.math.cumprod(tf.fill([max_power], q))
        q_powers = tf.concat([[tf.constant(1.0, dtype=tf.float64)], q_powers], axis=0)  # q^0, q^1, ..., q^(2*degree)

        askey_wilson_basis = [tf.ones_like(x, dtype=tf.float64)]

        if self.degree > 0:
            askey_wilson_basis.append(
                (2 * (1 + ab * q) * x - (a + b) * (1 + cd * q))
                / (1 + abcd * q_powers[2])
            )

        for n in range(2, self.degree + 1):
            q_n = q_powers[n]
            q_nm1 = q_powers[n - 1]
            q_2nm2 = q_powers[2 * n - 2]
            q_2nm1 = q_powers[2 * n - 1]
            q_2n = q_powers[2 * n]

            An = (
                (1 - ab * q_nm1)
                * (1 - cd * q_nm1)
                * (1 - abcd * q_2nm2)
            )
            An /= (1 - abcd * q_2nm1) * (1 - abcd * q_2n)
            Cn = (
                (1 - q_n)
                * (1 - ab * q_nm1)
                * (1 - cd * q_nm1)
                * (1 - abcd * q_2nm2)
            )
            Cn /= (1 - abcd * q_2nm2) * (1 - abcd * q_2nm1)
            askey_wilson_basis.append(
                ((2 * x - An) * askey_wilson_basis[n - 1] - Cn * askey_wilson_basis[n - 2]) / (1 - q_n)
            )

        return tf.cast(tf.stack(askey_wilson_basis, axis=-1), orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "a_init": self.a_init,
                "a_trainable": self.a_trainable,
                "b_init": self.b_init,
                "b_trainable": self.b_trainable,
                "c_init": self.c_init,
                "c_trainable": self.c_trainable,
                "d_init": self.d_init,
                "d_trainable": self.d_trainable,
                "q_init": self.q_init,
                "q_trainable": self.q_trainable,
            }
        )
        return config


__all__ = [
    "AskeyWilson",
]
