# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Wilson polynomials.

This module implements Wilson polynomials :math:`W_n(x^2; a, b, c, d)` which sit
at the top of the Askey scheme of hypergeometric orthogonal polynomials.

Mathematical Background
-----------------------
The Wilson polynomials satisfy the recurrence (DLMF 18.27.8), with :math:`t = x^2`:

.. math::
    W_0 &= 1 \\
    W_1 &= t - (A_0 - a^2) \\
    W_{n+1} &= (t - A_n - C_n + a^2) W_n - A_{n-1} C_n W_{n-1}

where

.. math::
    A_n &= \frac{(n+a+b)(n+a+c)(n+a+d)(n+a+b+c+d-1)}{(2n+a+b+c+d-1)(2n+a+b+c+d)} \\
    C_n &= \frac{n(n+b+c-1)(n+b+d-1)(n+c+d-1)}{(2n+a+b+c+d-2)(2n+a+b+c+d-1)}

Parameters must satisfy :math:`a, b, c, d > 0` for orthogonality.

References
----------
.. [DLMF] NIST Digital Library of Mathematical Functions, §18.27
   https://dlmf.nist.gov/18.27
"""

import warnings

import tensorflow as tf

from arnold.utils.compilation import kan_function
from arnold.utils.constraints import softplus_lower_bound
from arnold.utils.weights import create_bounded_param_logits

from ..poly_base import PolynomialBase


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="Wilson")
class Wilson(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using the Wilson polynomials :math:`W_n(x^2; a,b,c,d)`.

    Recurrence (DLMF 18.27.8), with ``x`` treated via :math:`t = x^2`:

    .. math::

        W_0 = 1 \\
        W_1 = t - (A_0 - a^2) \\
        W_{n+1} = \big(t - (A_n + C_n - a^2)\big) W_n - A_{n-1} C_n W_{n-1}

    where

    .. math::
        A_n = \frac{(n+a+b)(n+a+c)(n+a+d)(n+a+b+c+d-1)}{(2n+a+b+c+d-1)(2n+a+b+c+d)} \\
        C_n = \frac{n(n+b+c-1)(n+b+d-1)(n+c+d-1)}{(2n+a+b+c+d-2)(2n+a+b+c+d-1)}

    Parameters must satisfy ``a,b,c,d > 0`` for orthogonality; evaluation is
    performed in float64 for stability and cast back to the input dtype.
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        a_init: float | None = None,
        a_trainable=True,
        b_init: float | None = None,
        b_trainable=True,
        c_init: float | None = None,
        c_trainable=True,
        d_init: float | None = None,
        d_trainable=True,
        orthonormal: bool = False,
        input_clip=None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        degree : int
            Maximum polynomial degree.
        units : int
            Output dimensionality.
        a_init, b_init, c_init, d_init : float | None
            Initial values for Wilson parameters. Must be positive for orthogonality.
            Defaults to None (initialized via RandomNormal, then softplus-constrained).
        a_trainable, b_trainable, c_trainable, d_trainable : bool
            Trainability flags for each parameter. Default True.
        orthonormal : bool, default False
            When True, rescales :math:`W_n` by the orthonormalization factor
            derived from the squared norm integral. The normalization uses:

            .. math::

                h_n = \frac{n! \, \Gamma(n+a+b+c+d-1) \prod_{(p,q) \in \{(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)\}} \Gamma(n+p+q)}
                     {\Gamma(2n+a+b+c+d-1) \prod_{(p,q)} \Gamma(p+q)}

            and basis is divided by :math:`\sqrt{h_n}`.
        input_clip : tuple[float, float] | None
            Optional input clipping range.
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        self.orthonormal = orthonormal
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

        self.a_init = a_init
        self.a_trainable = a_trainable
        self.b_init = b_init
        self.b_trainable = b_trainable
        self.c_init = c_init
        self.c_trainable = c_trainable
        self.d_init = d_init
        self.d_trainable = d_trainable

        # Validate positivity of explicit init values for Wilson orthogonality
        for name, val in [("a", a_init), ("b", b_init), ("c", c_init), ("d", d_init)]:
            if val is not None and val <= 0:
                warnings.warn(
                    f"Wilson polynomial requires {name} > 0 for orthogonality, got {name}_init={val}. "
                    "The weight function is only well-defined for positive parameters.",
                    UserWarning,
                    stacklevel=2,
                )

        self.a_logits = None
        self.b_logits = None
        self.c_logits = None
        self.d_logits = None

        # Precompute orthonormal scaling factors if needed
        self._hn_sqrt_inv = None

    def build(self, input_shape):
        super().build(input_shape)
        # Use softplus-based parameterization for positivity constraint
        # Parameters are stored as logits, transformed via softplus + eps in forward pass
        self.a_logits = create_bounded_param_logits(
            self, "a", self.a_init, lower_bound=0.0, trainable=self.a_trainable
        )
        self.b_logits = create_bounded_param_logits(
            self, "b", self.b_init, lower_bound=0.0, trainable=self.b_trainable
        )
        self.c_logits = create_bounded_param_logits(
            self, "c", self.c_init, lower_bound=0.0, trainable=self.c_trainable
        )
        self.d_logits = create_bounded_param_logits(
            self, "d", self.d_init, lower_bound=0.0, trainable=self.d_trainable
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)

        # Use softplus-based positivity: maps R -> (eps, inf) with smooth gradients
        a = softplus_lower_bound(tf.cast(self.a_logits, tf.float64), lower_bound=0.0)
        b = softplus_lower_bound(tf.cast(self.b_logits, tf.float64), lower_bound=0.0)
        c = softplus_lower_bound(tf.cast(self.c_logits, tf.float64), lower_bound=0.0)
        d = softplus_lower_bound(tf.cast(self.d_logits, tf.float64), lower_bound=0.0)

        t = tf.square(x)  # t = x^2
        ones = tf.ones_like(t, dtype=tf.float64)

        def A(n_f):
            return (
                (n_f + a + b)
                * (n_f + a + c)
                * (n_f + a + d)
                * (n_f + a + b + c + d - 1.0)
                / ((2.0 * n_f + a + b + c + d - 1.0) * (2.0 * n_f + a + b + c + d))
            )

        def C(n_f):
            return (
                n_f
                * (n_f + b + c - 1.0)
                * (n_f + b + d - 1.0)
                * (n_f + c + d - 1.0)
                / ((2.0 * n_f + a + b + c + d - 2.0) * (2.0 * n_f + a + b + c + d - 1.0))
            )

        A0 = A(tf.constant(0.0, dtype=tf.float64))
        a2 = tf.square(a)

        W0 = ones
        if self.degree == 0:
            return tf.cast(tf.expand_dims(W0, -1), orig_dtype)

        W1 = t - (A0 - a2)
        if self.degree == 1:
            basis = tf.stack([W0, W1], axis=-1)
            return tf.cast(basis, orig_dtype)

        basis = [W0, W1]
        W_prev, W_curr, A_prev = W0, W1, A0
        for n in range(1, self.degree):
            n_f = tf.cast(n, tf.float64)
            A_n = A(n_f)
            C_n = C(n_f)
            W_next = (t - (A_n + C_n - a2)) * W_curr - (A_prev * C_n) * W_prev
            basis.append(W_next)
            W_prev, W_curr, A_prev = W_curr, W_next, A_n

        stacked = tf.stack(basis, axis=-1)

        if self.orthonormal:
            # Compute orthonormalization factors h_n for Wilson polynomials
            # h_n = n! * Gamma(n+a+b+c+d-1) * prod_{pairs} Gamma(n+p+q) /
            #       (Gamma(2n+a+b+c+d-1) * prod_{pairs} Gamma(p+q))
            # We compute log(h_n) for numerical stability, then exp(-0.5 * log_hn)
            log_hn = []
            for n in range(self.degree + 1):
                n_f = tf.cast(n, tf.float64)
                # log(n!)
                log_nfac = tf.math.lgamma(n_f + 1.0)
                # log(Gamma(n + a + b + c + d - 1))
                log_gamma_sum = tf.math.lgamma(n_f + a + b + c + d - 1.0)
                # Pair products: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
                log_pair_num = (
                    tf.math.lgamma(n_f + a + b) + tf.math.lgamma(n_f + a + c) +
                    tf.math.lgamma(n_f + a + d) + tf.math.lgamma(n_f + b + c) +
                    tf.math.lgamma(n_f + b + d) + tf.math.lgamma(n_f + c + d)
                )
                log_pair_den = (
                    tf.math.lgamma(a + b) + tf.math.lgamma(a + c) +
                    tf.math.lgamma(a + d) + tf.math.lgamma(b + c) +
                    tf.math.lgamma(b + d) + tf.math.lgamma(c + d)
                )
                # log(Gamma(2n + a + b + c + d - 1))
                log_gamma_2n = tf.math.lgamma(2.0 * n_f + a + b + c + d - 1.0)
                # log(h_n)
                log_h = log_nfac + log_gamma_sum + log_pair_num - log_gamma_2n - log_pair_den
                log_hn.append(log_h)
            # Scale factors: 1 / sqrt(h_n)
            log_hn_tensor = tf.stack(log_hn)  # (degree+1,)
            scale = tf.exp(-0.5 * log_hn_tensor)
            scale = tf.reshape(scale, (1, 1, -1))  # broadcast over (B, D, degree+1)
            stacked = stacked * tf.cast(scale, stacked.dtype)

        return tf.cast(stacked, orig_dtype)

    def clenshaw_basis(self, x: tf.Tensor) -> tf.Tensor:
        # Wilson pseudo_vandermonde is already recurrence-based; reuse it.
        return self.pseudo_vandermonde(x)

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
                "orthonormal": self.orthonormal,
            }
        )
        return config


__all__ = [
    "Wilson",
]
