# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Al-Salam-Carlitz polynomials.

This module implements the Al-Salam-Carlitz polynomials :math:`U_n^{(a)}(x; q)` and
:math:`V_n^{(a)}(x; q)`, which are q-analogs of the Hermite polynomials.

Mathematical Background
-----------------------
The Al-Salam-Carlitz polynomials of the first kind :math:`U_n^{(a)}(x; q)` satisfy:

.. math::
    U_{-1}^{(a)}(x; q) &= 0 \\
    U_0^{(a)}(x; q) &= 1 \\
    U_{n+1}^{(a)}(x; q) &= (x - (1+a)q^n) U_n^{(a)} + a q^{n-1}(1-q^n) U_{n-1}^{(a)}

The second kind :math:`V_n^{(a)}(x; q) = U_n^{(a)}(x; 1/q)`.

References
----------
.. [Wikipedia] https://en.wikipedia.org/wiki/Al-Salam%E2%80%93Carlitz_polynomials
.. [Core] https://core.ac.uk/download/pdf/82826366.pdf
"""

from abc import ABC, abstractmethod

import tensorflow as tf

from arnold.utils.compilation import kan_function
from arnold.utils.weights import create_trainable_param

from ..poly_base import PolynomialBase


tfk = tf.keras
kan_fn = kan_function()


@tfk.utils.register_keras_serializable(package="arnold", name="AlSalamCarlitz")
class AlSalamCarlitz(PolynomialBase, ABC):
    r"""
    Abstract base class for Kolmogorov-Arnold Network layer using Al-Salam-Carlitz polynomials.

    The Al-Salam–Carlitz polynomials $U^{(a)}_{n} (x;q)$ and $V^{(a)}_{n} (x;q)$ are two families of
    basic hypergeometric orthogonal polynomials.

    See also: https://en.wikipedia.org/wiki/Al-Salam%E2%80%93Carlitz_polynomials
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        a_init: float | None = None,
        a_trainable: bool = True,
        q_init: float | None = None,
        q_trainable: bool = True,
        input_clip=None,
        orthonormal: bool = False,
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
            Initial value for ``a`` (defaults to 0 when None).
        a_trainable : bool
            Whether ``a`` is trainable.
        q_init : float | None
            Initial value for ``q`` (defaults to 1 when None).
        q_trainable : bool
            Whether ``q`` is trainable.
        input_clip : tuple[float, float] | None
            Optional input clamp.
        orthonormal : bool, default False
            Placeholder for future support. Currently not implemented for Al-Salam–Carlitz.
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

        self.a_init = a_init
        self.a_trainable = a_trainable
        self.q_init = q_init
        self.q_trainable = q_trainable
        self.orthonormal = orthonormal
        self.a = None
        self.q = None

    def build(self, input_shape):
        super().build(input_shape)
        self.a = create_trainable_param(self, "a", self.a_init, initializer="zeros", trainable=self.a_trainable)
        self.q = create_trainable_param(self, "q", self.q_init, initializer="ones", trainable=self.q_trainable)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "a_init": self.a_init,
                "a_trainable": self.a_trainable,
                "q_init": self.q_init,
                "q_trainable": self.q_trainable,
                "orthonormal": self.orthonormal,
            }
        )
        return config

    @abstractmethod
    def pseudo_vandermonde(self, x):
        """Compute Al-Salam–Carlitz basis; implemented by subclasses."""
        raise NotImplementedError


@tfk.utils.register_keras_serializable(package="arnold", name="AlSalamCarlitz1st")
class AlSalamCarlitz1st(AlSalamCarlitz):
    r"""
    Kolmogorov-Arnold Network layer using the Al-Salam-Carlitz polynomials :math:`U^{(a)}_{n} (x;q)`.

    These polynomials satisfy the three-term recurrence relation

    * :math:`U^{(a)}_{-1} (x;q) = 0`
    * :math:`U^{(a)}_{0} (x;q) = 1`
    * :math:`U^{(a)}_{n+1} (x;q) = (x - (1 + a) q^{n}) U^{(a)}_{n} (x;q) + a q^{n-1} (1 - q^{n}) U^{(a)}_{n-1} (x;q)`

    See also: https://core.ac.uk/download/pdf/82826366.pdf

    Note
    ----
    This layer uses ``jit_compile=False`` due to XLA limitations with
    dynamic q-polynomial recurrences. Future versions may provide XLA support.
    """

    @kan_function(jit_compile=False)
    def pseudo_vandermonde(self, x):
        """Compute Al-Salam-Carlitz U basis."""
        # Cast trainable params to x.dtype to avoid dtype mismatch
        a = tf.cast(self.a, x.dtype)
        q = tf.cast(self.q, x.dtype)

        al_salam_carlitz_basis = [tf.ones_like(x)]  # U^{(a)}_0 = 1

        if self.degree > 0:
            al_salam_carlitz_basis.append(x - (1.0 + a))  # U^{(a)}_1 = x - (1 + a)

        if self.degree > 1:
            # Precompute q powers: [q^0, q^1, ..., q^{degree-1}]
            q_powers = tf.math.cumprod(tf.fill([self.degree], q))
            q_powers = tf.concat([[tf.ones_like(q)], q_powers[:-1]], axis=0)

            for n in range(1, self.degree):
                q_n = q_powers[n]
                q_nm1 = q_powers[n - 1]
                # U_{n+1} = (x - (1+a) q^n) U_n + a q^{n-1} (1 - q^n) U_{n-1}
                al_salam_carlitz_basis.append(
                    (x - (1.0 + a) * q_n) * al_salam_carlitz_basis[n]
                    + a * q_nm1 * (1.0 - q_n) * al_salam_carlitz_basis[n - 1]
                )

        return tf.stack(al_salam_carlitz_basis, axis=-1)

    def clenshaw_basis(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute Al-Salam-Carlitz U basis using tf.scan for XLA compatibility.

        This implementation uses tf.scan to accumulate polynomial values,
        making it compatible with XLA compilation via jit_compile=True.
        """
        a = tf.cast(self.a, x.dtype)
        q = tf.cast(self.q, x.dtype)
        degree = self.degree
        
        # Flatten x for processing
        x = tf.reshape(x, (-1, self.input_dim))  # (batch, input_dim)
        
        # U_0 = 1
        u0 = tf.ones_like(x)
        if degree == 0:
            return tf.expand_dims(u0, -1)
        
        # U_1 = x - (1 + a)
        u1 = x - (1.0 + a)
        if degree == 1:
            return tf.stack([u0, u1], axis=-1)
        
        # Use tf.scan for recurrence (n = 1, 2, ..., degree-1)
        # At step n, compute U_{n+1} from U_n and U_{n-1}
        def step(carry, n):
            u_n, u_n_1 = carry
            n_f = tf.cast(n, x.dtype)
            q_n = tf.pow(q, n_f)
            q_nm1 = tf.pow(q, n_f - 1.0)
            # U_{n+1} = (x - (1+a) q^n) U_n + a q^{n-1} (1 - q^n) U_{n-1}
            u_next = (x - (1.0 + a) * q_n) * u_n + a * q_nm1 * (1.0 - q_n) * u_n_1
            return (u_next, u_n)
        
        ns = tf.range(1, degree)  # n=1 gives U_2, n=degree-1 gives U_degree
        carries = tf.scan(step, ns, initializer=(u1, u0))
        # carries[0] has shape (degree-1, batch, input_dim)
        us = tf.transpose(carries[0], perm=[1, 2, 0])  # (batch, input_dim, degree-1)
        
        # Stack all basis functions
        basis = tf.concat([
            tf.expand_dims(u0, -1),
            tf.expand_dims(u1, -1),
            us
        ], axis=-1)
        
        return tf.reshape(basis, (-1, self.input_dim, degree + 1))


@tfk.utils.register_keras_serializable(package="arnold", name="AlSalamCarlitz2nd")
class AlSalamCarlitz2nd(AlSalamCarlitz):
    r"""
    Kolmogorov-Arnold Network layer using the Al-Salam-Carlitz polynomials :math:`V^{(a)}_{n} (x;q)`.

    There is a straightforward relationship between :math:`U^{(a)}_{n} (x;q)` and :math:`V^{(a)}_{n} (x;q)`:

    :math:`U^{(a)}_{n} (x; 1/q) = V^{(a)}_{n} (x;q)`

    See: Chihara, T.S. An Introduction to Orthogonal Polynomials; Mathematics Applied Series 13; Gordon and Breach: New York, NY, USA, 1978. Chapter VI, §10, pp. 195–198

    Note
    ----
    This layer uses ``jit_compile=False`` due to XLA limitations with
    dynamic q-polynomial recurrences. Future versions may provide XLA support.
    """

    @kan_function(jit_compile=False)
    def pseudo_vandermonde(self, x):
        """Compute Al-Salam-Carlitz V basis."""
        # Cast trainable params to x.dtype to avoid dtype mismatch
        a = tf.cast(self.a, x.dtype)
        q = tf.cast(self.q, x.dtype)

        al_salam_carlitz_basis = [tf.ones_like(x)]  # V^{(a)}_0 = 1

        if self.degree > 0:
            al_salam_carlitz_basis.append(x - (1.0 + a))  # V^{(a)}_1 = x - (1 + a)

        if self.degree > 1:
            # V uses (1/q) in place of q
            q_inv = 1.0 / q
            q_inv_powers = tf.math.cumprod(tf.fill([self.degree], q_inv))
            q_inv_powers = tf.concat([[tf.ones_like(q_inv)], q_inv_powers[:-1]], axis=0)

            for n in range(1, self.degree):
                q_n = q_inv_powers[n]
                q_nm1 = q_inv_powers[n - 1]
                # Same recurrence as U, but with 1/q
                al_salam_carlitz_basis.append(
                    (x - (1.0 + a) * q_n) * al_salam_carlitz_basis[n]
                    + a * q_nm1 * (1.0 - q_n) * al_salam_carlitz_basis[n - 1]
                )

        return tf.stack(al_salam_carlitz_basis, axis=-1)

    def clenshaw_basis(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute Al-Salam-Carlitz V basis using tf.scan for XLA compatibility.

        Uses the relationship V(x; q) = U(x; 1/q) with XLA-compatible tf.scan.
        """
        a = tf.cast(self.a, x.dtype)
        q = tf.cast(self.q, x.dtype)
        q_inv = 1.0 / q  # V uses inverse q
        degree = self.degree
        
        # Flatten x for processing
        x = tf.reshape(x, (-1, self.input_dim))  # (batch, input_dim)
        
        # V_0 = 1
        v0 = tf.ones_like(x)
        if degree == 0:
            return tf.expand_dims(v0, -1)
        
        # V_1 = x - (1 + a)
        v1 = x - (1.0 + a)
        if degree == 1:
            return tf.stack([v0, v1], axis=-1)
        
        # Use tf.scan for recurrence
        def step(carry, n):
            v_n, v_n_1 = carry
            n_f = tf.cast(n, x.dtype)
            q_n = tf.pow(q_inv, n_f)
            q_nm1 = tf.pow(q_inv, n_f - 1.0)
            # Same recurrence as U, with 1/q
            v_next = (x - (1.0 + a) * q_n) * v_n + a * q_nm1 * (1.0 - q_n) * v_n_1
            return (v_next, v_n)
        
        ns = tf.range(1, degree)
        carries = tf.scan(step, ns, initializer=(v1, v0))
        vs = tf.transpose(carries[0], perm=[1, 2, 0])
        
        basis = tf.concat([
            tf.expand_dims(v0, -1),
            tf.expand_dims(v1, -1),
            vs
        ], axis=-1)
        
        return tf.reshape(basis, (-1, self.input_dim, degree + 1))


__all__ = [
    "AlSalamCarlitz",
    "AlSalamCarlitz1st",
    "AlSalamCarlitz2nd",
]
