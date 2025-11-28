# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Charlier polynomials.

This module implements Charlier polynomials :math:`C_n(x; a)`, a family of
discrete orthogonal polynomials on the non-negative integers.

Mathematical Background
-----------------------
Charlier polynomials are orthogonal with respect to the Poisson distribution
with parameter :math:`a > 0`. They satisfy the recurrence:

.. math::

    C_{n+1}(x; a) = (x - n - a) C_n(x; a) - n a C_{n-1}(x; a)

with :math:`C_0(x; a) = 1` and :math:`C_1(x; a) = x - a`.

The parameter :math:`a` corresponds to the mean of the Poisson distribution.

References
----------
.. [DLMF] NIST Digital Library of Mathematical Functions, §18.19
   https://dlmf.nist.gov/18.19
"""

import tensorflow as tf

from arnold.utils.compilation import kan_function
from arnold.utils.constants import PARAM_EPS
from arnold.utils.weights import create_trainable_param

from ..poly_base import PolynomialBase


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="Charlier")
class Charlier(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Charlier polynomials.

    Charlier polynomials :math:`C_n(x; a)` are orthogonal on the non-negative
    integers with respect to the Poisson weight :math:`\frac{a^x e^{-a}}{x!}`.

    The parameter :math:`a > 0` is enforced via softplus transformation for
    numerical stability.

    Attributes
    ----------
    a : tf.Variable
        Poisson parameter :math:`a > 0`.
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        a_init: float | None = None,
        a_trainable: bool = True,
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
        a_init : float | None
            Initial value for ``a``; defaults to RandomNormal when None.
        a_trainable : bool
            Trainability flag.
        input_clip : tuple[float, float] | None
            Optional input clamp.
        orthonormal : bool, default False
            If True, scale basis by :math:`1/\sqrt{h_n}` where
            :math:`h_n = n! a^n` is the squared norm.
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        self.orthonormal = kwargs.pop("orthonormal", False)
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

        self.a_init = a_init
        self.a_trainable = a_trainable

        self.a = None

    def build(self, input_shape):
        super().build(input_shape)

        self.a = create_trainable_param(self, "a", self.a_init, trainable=self.a_trainable)

    @kan_fn
    def pseudo_vandermonde(self, x):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)

        # Enforce a > 0 via softplus for numerical stability
        a_raw = tf.cast(self.a, tf.float64)
        a = tf.nn.softplus(a_raw) + PARAM_EPS

        charlier_basis = [tf.ones_like(x, dtype=tf.float64)]

        if self.degree > 0:
            p1 = x - a
            charlier_basis.append(p1)

        for n in range(1, self.degree):
            # Recurrence: C_{n+1}(x) = (x - n - a) C_n(x) - n * a * C_{n-1}(x)
            n_float = tf.constant(float(n), dtype=tf.float64)
            p_new = (x - n_float - a) * charlier_basis[n] - n_float * a * charlier_basis[n - 1]
            charlier_basis.append(p_new)

        basis = tf.stack(charlier_basis, axis=-1)

        if self.orthonormal:
            # h_n = n! * a^n
            # Orthonormal scaling: divide by sqrt(h_n)
            n_vals = tf.range(self.degree + 1, dtype=tf.float64)
            log_factorial = tf.math.lgamma(n_vals + 1.0)  # log(n!)
            log_a_power = n_vals * tf.math.log(a + PARAM_EPS)  # n * log(a)
            log_h_n = log_factorial + log_a_power
            scale_factors = tf.exp(-0.5 * log_h_n)
            basis = basis * scale_factors

        return tf.cast(basis, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "a_init": self.a_init,
                "a_trainable": self.a_trainable,
                "orthonormal": self.orthonormal,
            }
        )
        return config


__all__ = [
    "Charlier",
]
