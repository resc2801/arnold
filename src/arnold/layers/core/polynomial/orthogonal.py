## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
import warnings
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from arnold.utils.compilation import kan_function
from arnold.utils.constants import PARAM_EPS
from arnold.utils.constraints import softplus_lower_bound
from arnold.utils.numerics import safe_acos
from arnold.utils.weights import create_bounded_param_logits, create_trainable_param

from .poly_base import PolynomialBase


tfk = tf.keras
tfkl = tfk.layers
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


@tfk.utils.register_keras_serializable(package="arnold", name="BannaiIto")
class BannaiIto(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using the Bannai-Ito polynomials.
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
        a_init, b_init, c_init : float | None
            Initial values for parameters; default to zeros when None.
        a_trainable, b_trainable, c_trainable : bool
            Trainability flags.
        input_clip : tuple[float, float] | None
            Optional input clamp.
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

        self.a = None
        self.b = None
        self.c = None

    def build(self, input_shape):
        super().build(input_shape)
        self.a = create_trainable_param(self, "a", self.a_init, initializer="zeros", trainable=self.a_trainable)
        self.b = create_trainable_param(self, "b", self.b_init, initializer="zeros", trainable=self.b_trainable)
        self.c = create_trainable_param(self, "c", self.c_init, initializer="zeros", trainable=self.c_trainable)

    @kan_fn
    def pseudo_vandermonde(self, x):
        # Cast trainable params to x.dtype to avoid dtype mismatch
        a = tf.cast(self.a, x.dtype)
        b = tf.cast(self.b, x.dtype)
        c = tf.cast(self.c, x.dtype)

        bannai_ito_basis = [tf.ones_like(x)]

        if self.degree > 0:
            bannai_ito_basis.append((x - a) / (b + c + 1.0))

        for n in range(2, self.degree + 1):
            An = (2 * n + b + c - 1) * (2 * n + b + c) / (2 * (n + b + c))
            Cn = -(n + b - 1) * (n + c - 1) / (2 * (n + b + c))

            bannai_ito_basis.append(
                ((x - An) * bannai_ito_basis[n - 1] - Cn * bannai_ito_basis[n - 2]) / (n + b + c)
            )

        return tf.stack(bannai_ito_basis, axis=-1)

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
            }
        )
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="Bessel")
class Bessel(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Bessel polynomials.

    Domain is :math:`\mathbb{R}`; coefficients grow rapidly and can overflow for
    large degrees in float32. Consider limiting ``degree`` or using float64 for
    high-order expansions.

    The Bessel polynomials are generated by the three-term recurrence relation:

    * :math:`y_{0}(x) = 1`
    * :math:`y_{1}(x) = x + 1`
    * :math:`y_{n}(x) = (2n - 1) * x * y_{n-1}(x) + y_{n-2}(x)` when n >= 2

    See also: https://en.wikipedia.org/wiki/Bessel_polynomials#Recursion
    """

    def __init__(self, degree: int, *, units: int, **kwargs):
        if degree > 15:
            warnings.warn(
                f"Bessel polynomials with degree={degree} > 15 may overflow in float32. "
                "Consider using promote_to_float64=True or reducing degree.",
                UserWarning,
                stacklevel=2,
            )
        super().__init__(degree=degree, units=units, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        # :math:`y_{0}(x) = 1`
        bessel_basis = [tf.ones_like(x)]

        if self.degree > 0:
            # :math:`y_{1}(x) = x + 1`
            bessel_basis.append(x + 1.0)

        for n in range(2, self.degree + 1):
            # :math:`y_{n}(x) = (2n - 1) * x * y_{n-1}(x) + y_{n-2}(x)` when n >= 2
            bessel_basis.append((2 * n - 1) * x * bessel_basis[n - 1] + bessel_basis[n - 2])

        return tf.stack(bessel_basis, axis=-1)

    def clenshaw_basis(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.reshape(x, (-1, self.input_dim))
        y0 = tf.ones_like(x)
        if self.degree == 0:
            return tf.expand_dims(y0, -1)
        y1 = x + 1.0
        if self.degree == 1:
            return tf.stack([y0, y1], axis=-1)

        def step(carry, n):
            yn_1, yn_2 = carry
            n_f = tf.cast(n, x.dtype)
            yn = (2.0 * n_f - 1.0) * x * yn_1 + yn_2
            return (yn, yn_1)  # Only return new carry

        ns = tf.range(2, self.degree + 1)
        carries = tf.scan(step, ns, initializer=(y1, y0))
        # carries[0] has shape (degree-1, batch, input_dim) - transpose to (batch, input_dim, degree-1)
        ys = tf.transpose(carries[0], perm=[1, 2, 0])
        basis = tf.concat([tf.expand_dims(y0, -1), tf.expand_dims(y1, -1), ys], axis=-1)
        return tf.reshape(basis, (-1, self.input_dim, self.degree + 1))


@tfk.utils.register_keras_serializable(package="arnold", name="Charlier")
class Charlier(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using the Charlier polynomials.

    The Charlier polynomials are generated by the three-term recurrence relation:

    * :math:`C_{-1; a}(x) = 0`
    * :math:`C_{0; a}(x) = 1`
    * :math:`x * C_{n}(x; a) = C_{n+1}(x; a) + (n + a) * C_{n}(x; a) + a * n * C_{n-1}(x; a)`

    for :math:`a>0`.

    The parameter ``a`` is stored as logits and mapped to ``(0, ∞)`` via
    ``softplus`` for smooth gradient flow near the constraint boundary.

    See: https://arxiv.org/pdf/1901.06041, eq. 1.4
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
            Initial value for ``a`` (must be >0); defaults to RandomNormal logits when None.
        a_trainable : bool
            Whether ``a`` is trainable.
        input_clip : tuple[float, float] | None
            Optional input clamp.
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

        self.a_init = a_init
        self.a_trainable = a_trainable
        self.a_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        # Use softplus-based parameterization for positivity constraint (consistent with other layers)
        self.a_logits = create_bounded_param_logits(
            self, "a", self.a_init, lower_bound=0.0, trainable=self.a_trainable
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        # :math:`a > 0` via softplus for smooth gradients (consistent with Wilson, RBFs, etc.)
        a = softplus_lower_bound(tf.cast(self.a_logits, x.dtype), lower_bound=0.0)

        # :math:`C_{-1; a}(x) = 0`
        # :math:`C_{0; a}(x) = 1`
        charlier_basis = [tf.ones_like(x)]

        if self.degree > 0:
            # :math`C_{1}(x; a) = (x - (n + a)) * C_{0}(x; a) - a*n*C_{-1}(x; a)`
            charlier_basis.append(x - (1 + a))

        for n in range(2, self.degree + 1):
            # :math:`x * C_{n}(x; a) = C_{n+1}(x; a) + (n + a) * C_{n}(x; a) + a * n * C_{n-1}(x; a)`
            charlier_basis.append((x - (n + a)) * charlier_basis[n - 1] - a * n * charlier_basis[n - 2])

        return tf.stack(charlier_basis, axis=-1)

    def clenshaw_basis(self, x: tf.Tensor) -> tf.Tensor:
        a = softplus_lower_bound(tf.cast(self.a_logits, x.dtype), lower_bound=0.0)
        x = tf.reshape(x, (-1, self.input_dim))
        C0 = tf.ones_like(x)
        if self.degree == 0:
            return tf.expand_dims(C0, -1)
        C1 = x - (1.0 + a)
        if self.degree == 1:
            return tf.stack([C0, C1], axis=-1)

        def step(carry, n):
            Cn_1, Cn_2 = carry
            n_f = tf.cast(n, x.dtype)
            Cn = (x - (n_f + a)) * Cn_1 - a * n_f * Cn_2
            return (Cn, Cn_1)  # Only return new carry

        ns = tf.range(2, self.degree + 1)
        carries = tf.scan(step, ns, initializer=(C1, C0))
        # carries[0] has shape (degree-1, batch, input_dim) - transpose to (batch, input_dim, degree-1)
        Cs = tf.transpose(carries[0], perm=[1, 2, 0])
        basis = tf.concat([tf.expand_dims(C0, -1), tf.expand_dims(C1, -1), Cs], axis=-1)
        return tf.reshape(basis, (-1, self.input_dim, self.degree + 1))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "a_init": self.a_init,
                "a_trainable": self.a_trainable,
            }
        )
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="Chebyshev")
class Chebyshev(PolynomialBase, ABC):
    r"""
    Abstract base class for Kolmogorov-Arnold Network layer using Chebyshev polynomial basis.

    TODO: check https://www.mathematik.uni-kassel.de/~koepf/Publikationen/cheby.pdf
    """

    def __init__(self, *args, **kwargs):
        r"""
        :param input_dim: This layers input size
        :type input_dim: int

        :param output_dim: This layers output size
        :type output_dim: int

        :param degree: The maximum degree of the polynomial basis element (default is 3).
        :type degree: int

        :param decompose_weights: Whether or not to represent the polynomial_coefficients weights tensor as a learnable Tucker decomposition. Default to False.
        :type decompose_weights: bool

        :param core_ranks: A 3-tuple of non-zero, positive integers giving the ranks of the Tucker decomposition core tensor. Ignored if `decompose_weights` is False; defaults to None.
        :type core_ranks: None | Tuple[int, int, int]

        :param tanh_x: Flag indicating whether to normalize any input to [-1, 1] using tanh before further processing.
        :type tanh_x: bool
        """
        super().__init__(*args, **kwargs)
        self.arange = tf.constant(tf.range(0, self.degree + 1, 1, dtype=float))

    @abstractmethod
    def pseudo_vandermonde(self, x):
        """Compute Chebyshev basis; implemented by subclasses."""
        raise NotImplementedError


@tfk.utils.register_keras_serializable(package="arnold", name="Chebyshev1st")
class Chebyshev1st(Chebyshev):
    r"""
    Chebyshev polynomials of the first kind in trigonometric form.

    .. math::

        T_n(x) = \cos(n \arccos(x)), \quad |x| \le 1

    Inputs are clipped to the canonical domain ``[-1, 1]`` by default.
    """

    def __init__(self, degree: int, *, units: int, input_clip: tuple[float, float] | None = (-1.0, 1.0), **kwargs):
        """
        Parameters
        ----------
        degree : int
            Maximum polynomial degree.
        units : int
            Output dimensionality.
        input_clip : tuple[float, float] | None
            Interval to clamp inputs before evaluating ``acos`` (defaults to [-1, 1]).
        orthonormal : bool, default False
            When True, rescales :math:`T_n` by :math:`\\sqrt{2/\\pi}` (and :math:`1/\\sqrt{\\pi}` for ``n=0``)
            to make the basis orthonormal with respect to weight :math:`1/\\sqrt{1-x^2}` on ``[-1,1]``.
        **kwargs :
            Forwarded to :class:`PolynomialBase` (e.g., activation, use_bias).
        """
        self.orthonormal = kwargs.pop("orthonormal", False)
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        # Use broadcasting instead of tf.tile for better performance
        x = tf.reshape(x, (-1, self.input_dim, 1))  # (B, D, 1)
        theta = safe_acos(x)  # (B, D, 1)
        # arange broadcasts: (1, 1, degree+1) * (B, D, 1) -> (B, D, degree+1)
        basis = tf.math.cos(theta * self.arange)
        if self.orthonormal:
            n = tf.range(0, self.degree + 1, dtype=basis.dtype)
            scale = tf.where(n == 0, tf.sqrt(1.0 / tf.constant(np.pi, dtype=basis.dtype)), tf.sqrt(2.0 / tf.constant(np.pi, dtype=basis.dtype)))
            basis = basis * scale
        return basis

    def clenshaw_basis(self, x: tf.Tensor) -> tf.Tensor:
        theta = safe_acos(tf.reshape(x, (-1, self.input_dim)))
        cos_theta = tf.cos(theta)
        T0 = tf.ones_like(theta)
        if self.degree == 0:
            return tf.expand_dims(T0, axis=-1)
        T1 = cos_theta
        if self.degree == 1:
            return tf.stack([T0, T1], axis=-1)

        def step(carry, _n):
            Tn_1, Tn_2 = carry
            Tn = 2.0 * cos_theta * Tn_1 - Tn_2
            return (Tn, Tn_1)  # Only return new carry

        ns = tf.range(2, self.degree + 1)
        carries = tf.scan(step, ns, initializer=(T1, T0))
        # carries[0] has shape (degree-1, batch, input_dim) - transpose to (batch, input_dim, degree-1)
        Ts = tf.transpose(carries[0], perm=[1, 2, 0])
        basis = tf.concat([tf.expand_dims(T0, -1), tf.expand_dims(T1, -1), Ts], axis=-1)
        if self.orthonormal:
            n = tf.range(0, self.degree + 1, dtype=basis.dtype)
            scale = tf.where(n == 0, tf.sqrt(1.0 / tf.constant(np.pi, dtype=basis.dtype)), tf.sqrt(2.0 / tf.constant(np.pi, dtype=basis.dtype)))
            basis = basis * scale
        return tf.reshape(basis, (-1, self.input_dim, self.degree + 1))

    def true_clenshaw_eval(self, x: tf.Tensor, coefficients: tf.Tensor) -> tf.Tensor:
        """
        True Clenshaw summation for Chebyshev polynomials with O(1) memory.

        Fuses basis computation with coefficient contraction using the Clenshaw
        recurrence for Chebyshev polynomials:

        .. math::

            b_{n+1} = 0, \\quad b_n = c_n \\\\
            b_{k-1} = c_{k-1} + 2x \\cdot b_k - b_{k+1} \\\\
            T_0 c_0 + T_1 c_1 + \\cdots + T_n c_n = b_0 - x \\cdot b_1

        This avoids materializing the full basis tensor, critical for high-degree
        polynomials on memory-constrained accelerators.

        :param x: Input tensor of shape (batch, input_dim).
        :param coefficients: Coefficient tensor of shape (input_dim, degree+1, output_dim).
        :returns: Output tensor of shape (batch, output_dim).
        """
        # Reshape x for computation: (batch, input_dim)
        x = tf.reshape(x, (-1, self.input_dim))  # (B, D)
        cos_x = tf.cos(safe_acos(x))  # Use acos for Chebyshev domain
        
        # coefficients: (input_dim, degree+1, output_dim)
        degree = self.degree
        
        # Apply orthonormal scaling to coefficients if needed
        if self.orthonormal:
            n = tf.range(0, degree + 1, dtype=coefficients.dtype)
            scale = tf.where(
                n == 0,
                tf.sqrt(1.0 / tf.constant(np.pi, dtype=coefficients.dtype)),
                tf.sqrt(2.0 / tf.constant(np.pi, dtype=coefficients.dtype))
            )
            # scale: (degree+1,) -> (1, degree+1, 1) for broadcasting
            coefficients = coefficients * tf.reshape(scale, (1, -1, 1))
        
        # Clenshaw recurrence: b_{k-1} = c_{k-1} + 2*cos_x*b_k - b_{k+1}
        # Start with b_{n+1} = 0, b_n = c_n
        # coefficients[:, k, :] is c_k for all input dims and output dims
        
        # Initialize: b_{n+1} = 0, b_n = c_n
        # Shape: (B, D, O) where O is output_dim
        b_next = tf.zeros((tf.shape(x)[0], self.input_dim, tf.shape(coefficients)[-1]), dtype=coefficients.dtype)
        b_curr = tf.broadcast_to(
            tf.expand_dims(coefficients[:, degree, :], 0),  # (1, D, O)
            (tf.shape(x)[0], self.input_dim, tf.shape(coefficients)[-1])
        )
        
        # cos_x for Chebyshev: (B, D) -> (B, D, 1) for broadcasting
        cos_x_expanded = tf.expand_dims(cos_x, -1)
        
        def clenshaw_step(carry, k):
            b_curr, b_next = carry
            # c_{k-1} + 2*cos_x*b_k - b_{k+1}
            c_k = coefficients[:, k, :]  # (D, O) -> broadcast to (B, D, O)
            c_k = tf.broadcast_to(tf.expand_dims(c_k, 0), tf.shape(b_curr))
            b_new = c_k + 2.0 * cos_x_expanded * b_curr - b_next
            return (b_new, b_curr)
        
        # Run recurrence from k = degree-1 down to k = 0
        if degree > 0:
            ks = tf.range(degree - 1, -1, -1)  # degree-1, degree-2, ..., 0
            (b_0, b_1) = tf.foldl(clenshaw_step, ks, initializer=(b_curr, b_next))
        else:
            b_0, b_1 = b_curr, b_next
        
        # Final result: b_0 - cos_x * b_1
        result = b_0 - cos_x_expanded * b_1
        
        # Sum over input dimensions: (B, D, O) -> (B, O)
        output = tf.reduce_sum(result, axis=1)
        
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"orthonormal": self.orthonormal})
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="Chebyshev2nd")
class Chebyshev2nd(Chebyshev):
    r"""
    Kolmogorov-Arnold Network layer using 2nd kind Chebyshev polynomials in trigonometric formulation

    .. math::
        :nowrap:

        \begin{equation}
        U_{n}(x) = \frac{\sin((n+1) * \arccos(x))}{\sin(\arccos(x))}, \, \lvert x \rvert \leq 1
        \end{equation}

    See: https://core.ac.uk/download/pdf/82763706.pdf

    Stable on ``[-1, 1]``; inputs are clipped by default to avoid ``acos``/``sin``
    singularities at the boundaries.
    """

    def __init__(self, degree: int, *, units: int, input_clip: tuple[float, float] | None = (-1.0, 1.0), orthonormal: bool = False, **kwargs):
        """
        Parameters
        ----------
        degree : int
            Maximum polynomial degree.
        units : int
            Output dimensionality.
        input_clip : tuple[float, float] | None
            Interval to clamp inputs before evaluating ``acos`` (defaults to [-1, 1]).
        orthonormal : bool, default False
            When True, rescales :math:`U_n` by :math:`\\sqrt{2/\\pi}` (orthonormal with respect to weight ``sqrt(1-x^2)`` on ``[-1,1]``).
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        self.orthonormal = orthonormal
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        # Use broadcasting instead of tf.tile for better performance
        x = tf.reshape(x, (-1, self.input_dim, 1))  # (B, D, 1)
        theta = safe_acos(x)  # (B, D, 1)
        k = tf.cast(tf.range(1, self.degree + 2), theta.dtype)  # 1..degree+1
        k = tf.reshape(k, (1, 1, -1))  # (1, 1, degree+1)
        # Broadcasting: (B, D, 1) * (1, 1, degree+1) -> (B, D, degree+1)
        num = tf.math.sin(theta * k)
        den = tf.math.sin(theta)
        is_boundary = tf.abs(den) < 1e-8
        sign = tf.where(x >= 0, tf.ones_like(x), -tf.ones_like(x))
        boundary_val = tf.pow(sign, k - 1.0) * k
        safe_den = tf.where(is_boundary, tf.ones_like(den), den)
        normal = num / safe_den
        basis = tf.where(is_boundary, boundary_val, normal)
        if self.orthonormal:
            scale = tf.sqrt(2.0 / tf.constant(np.pi, dtype=basis.dtype))
            basis = basis * scale
        return basis

    def clenshaw_basis(self, x: tf.Tensor) -> tf.Tensor:
        theta = safe_acos(tf.reshape(x, (-1, self.input_dim)))
        U0 = tf.ones_like(theta)
        if self.degree == 0:
            return tf.expand_dims(U0, -1)
        U1 = 2.0 * tf.cos(theta)
        if self.degree == 1:
            return tf.stack([U0, U1], axis=-1)

        def step(carry, _n):
            Un_1, Un_2 = carry
            Un = 2.0 * tf.cos(theta) * Un_1 - Un_2
            return (Un, Un_1)  # Only return new carry

        ns = tf.range(2, self.degree + 1)
        carries = tf.scan(step, ns, initializer=(U1, U0))
        # carries[0] has shape (degree-1, batch, input_dim) - transpose to (batch, input_dim, degree-1)
        Us = tf.transpose(carries[0], perm=[1, 2, 0])
        basis = tf.concat([tf.expand_dims(U0, -1), tf.expand_dims(U1, -1), Us], axis=-1)
        if self.orthonormal:
            scale = tf.sqrt(2.0 / tf.constant(np.pi, dtype=basis.dtype))
            basis = basis * scale
        return tf.reshape(basis, (-1, self.input_dim, self.degree + 1))

    def get_config(self):
        config = super().get_config()
        config.update({"orthonormal": self.orthonormal})
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="Chebyshev3rd")
class Chebyshev3rd(Chebyshev):
    r"""
    Kolmogorov-Arnold Network layer using 3rd kind Chebyshev polynomials in trigonometric formulation

    .. math::
        :nowrap:

        \begin{equation}
        V_{n}(x) = \frac{\cos((n + \tfrac{1}{2}) * \arccos(x))}{\cos(\tfrac{1}{2} * \arccos(x))}, \, \lvert x \rvert \leq 1
        \end{equation}

    See: https://core.ac.uk/download/pdf/82763706.pdf

    Stable on ``[-1, 1]``; inputs are clipped by default to avoid ``acos``/``cos``
    singularities at the boundaries.
    """

    def __init__(self, degree: int, *, units: int, input_clip: tuple[float, float] | None = (-1.0, 1.0), **kwargs):
        """
        Parameters
        ----------
        degree : int
            Maximum polynomial degree.
        units : int
            Output dimensionality.
        input_clip : tuple[float, float] | None
            Interval to clamp inputs before evaluating ``acos`` (defaults to [-1, 1]).
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        # Use broadcasting instead of tf.tile for better performance
        x = tf.reshape(x, (-1, self.input_dim, 1))  # (B, D, 1)
        theta = safe_acos(x)  # (B, D, 1)
        k = tf.range(0, self.degree + 1, dtype=theta.dtype) + 0.5  # (degree+1,)
        k = tf.reshape(k, (1, 1, -1))  # (1, 1, degree+1)
        # Broadcasting: (B, D, 1) * (1, 1, degree+1) -> (B, D, degree+1)
        num = tf.math.cos(theta * k)
        den = tf.math.cos(0.5 * theta)
        den = tf.where(tf.abs(den) < 1e-8, tf.ones_like(den), den)
        return num / den


@tfk.utils.register_keras_serializable(package="arnold", name="Chebyshev4th")
class Chebyshev4th(Chebyshev):
    r"""
    Kolmogorov-Arnold Network layer using 4th kind Chebyshev polynomials in trigonometric formulation

    .. math::
        :nowrap:

        \begin{equation}
        W_{n}(x) = \frac{\sin((n + \tfrac{1}{2}) * \arccos(x))}{\sin(\tfrac{1}{2} * \arccos(x))}, \, \lvert x \rvert \leq 1
        \end{equation}

    See: https://core.ac.uk/download/pdf/82763706.pdf

    Stable on ``[-1, 1]``; inputs are clipped by default to avoid ``acos``/``sin``
    singularities at the boundaries.
    """

    def __init__(self, degree: int, *, units: int, input_clip: tuple[float, float] | None = (-1.0, 1.0), **kwargs):
        """
        Parameters
        ----------
        degree : int
            Maximum polynomial degree.
        units : int
            Output dimensionality.
        input_clip : tuple[float, float] | None
            Interval to clamp inputs before evaluating ``acos`` (defaults to [-1, 1]).
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        # Use broadcasting instead of tf.tile for better performance
        x = tf.reshape(x, (-1, self.input_dim, 1))  # (B, D, 1)
        theta = safe_acos(x)  # (B, D, 1)
        k = tf.range(0, self.degree + 1, dtype=theta.dtype) + 0.5  # (degree+1,)
        k = tf.reshape(k, (1, 1, -1))  # (1, 1, degree+1)
        # Broadcasting: (B, D, 1) * (1, 1, degree+1) -> (B, D, degree+1)
        num = tf.math.sin(theta * k)
        den = tf.math.sin(0.5 * theta)
        den = tf.where(tf.abs(den) < 1e-8, tf.ones_like(den), den)
        return num / den


@tfk.utils.register_keras_serializable(package="arnold", name="Gegenbauer")
class Gegenbauer(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Gegenbauer polynomials.

    The Gegenbauer polynomials are generated by the three-term recurrence relation:
    valid for ``alpha > -0.5`` with inputs on ``[-1, 1]`` (inputs are clipped by default).

    * :math:`C^{\alpha}_{0}(x) = 1`
    * :math:`C^{\alpha}_{1}(x) = 2 * \alpha * x`
    * :math:`C^{\alpha}_{n+1}(x) = \frac{(2 * (n + \alpha) * x * C^{\alpha}_{n}(x)) - ((n + 2 * \alpha - 1) * C^{\alpha}_{n - 1}(x))}{n + 1}` when n >= 1

    See also: https://en.wikipedia.org/wiki/Gegenbauer_polynomials#Characterizations

    They generalize Legendre polynomials and Chebyshev polynomials, and are special cases of Jacobi polynomials.
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        alpha_init: float | None = None,
        alpha_trainable: bool = True,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
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
        alpha_init : float | None
            Initial value for ``alpha``; must satisfy ``alpha > -0.5``; defaults to RandomNormal when None.
        alpha_trainable : bool
            Whether ``alpha`` is trainable.
        input_clip : tuple[float, float] | None
            Optional input clamp (defaults to [-1, 1]).
        orthonormal : bool, default False
            When True, rescales :math:`C_n^{\\alpha}` to be orthonormal with respect to
            weight :math:`(1-x^2)^{\\alpha-1/2}` on ``[-1,1]``.
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

        self.alpha_init = alpha_init
        self.alpha_trainable = alpha_trainable
        self.orthonormal = orthonormal
        self.alpha_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        self.alpha_logits = create_bounded_param_logits(
            self, "alpha", self.alpha_init, lower_bound=-0.5, trainable=self.alpha_trainable
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        # :math:`C^{\alpha}_{0}(x) = 1`
        # softplus_lower_bound ensures alpha > -0.5 with smooth gradients
        alpha = softplus_lower_bound(tf.cast(self.alpha_logits, x.dtype), lower_bound=-0.5)
        gegenbauer_basis = [tf.ones_like(x)]

        if self.degree > 0:
            # :math:`C^{\alpha}_{1}(x) = 2 * \alpha x`
            gegenbauer_basis.append(2 * alpha * x)

        for n in range(2, self.degree + 1):
            # :math:`C^{\alpha}_{n+1}(x) = \frac{(2 * (n + \alpha) * x * C^{\alpha}_{n}(x)) - ((n + 2 * \alpha - 1) * C^{\alpha}_{n - 1}(x))}{n + 1}` when n >= 1
            gegenbauer_basis.append(
                (
                    (2 * ((n - 1) + alpha) * x * gegenbauer_basis[n - 1])
                    - (((n - 1) + 2 * alpha - 1) * gegenbauer_basis[n - 2])
                )
                / n
            )

        basis = tf.stack(gegenbauer_basis, axis=-1)
        if self.orthonormal:
            # norm = π 2^{1-2α} Γ(n+2α)/(n!(n+α) Γ(α)^2)
            n = tf.range(0, self.degree + 1, dtype=basis.dtype)
            alpha = tf.cast(alpha, basis.dtype)
            log_norm = (
                tf.math.log(np.pi)
                + (1.0 - 2.0 * alpha) * tf.math.log(2.0)
                + tf.math.lgamma(n + 2.0 * alpha)
                - tf.math.lgamma(n + 1.0)
                - tf.math.log(n + alpha)
                - 2.0 * tf.math.lgamma(alpha)
            )
            scale = tf.exp(-0.5 * log_norm)
            basis = basis * scale
        return basis

    def clenshaw_basis(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.reshape(x, (-1, self.input_dim))
        C0 = tf.ones_like(x)
        if self.degree == 0:
            return tf.expand_dims(C0, -1)
        # softplus_lower_bound ensures alpha > -0.5 with smooth gradients
        alpha = softplus_lower_bound(tf.cast(self.alpha_logits, x.dtype), lower_bound=-0.5)
        C1 = 2.0 * alpha * x
        if self.degree == 1:
            return tf.stack([C0, C1], axis=-1)

        def step(carry, n):
            Cn_1, Cn_2 = carry
            n_f = tf.cast(n, x.dtype)
            # Match pseudo_vandermonde: for n=2,3,... computing C_n from C_{n-1}, C_{n-2}
            # C_n = (2*(n-1+alpha)*x*C_{n-1} - (n-1+2*alpha-1)*C_{n-2}) / n
            Cn = ((2.0 * (n_f - 1.0 + alpha) * x * Cn_1) - ((n_f - 1.0 + 2.0 * alpha - 1.0) * Cn_2)) / n_f
            return (Cn, Cn_1)  # Only return new carry

        ns = tf.range(2, self.degree + 1)
        carries = tf.scan(step, ns, initializer=(C1, C0))
        # carries[0] has shape (degree-1, batch, input_dim) - transpose to (batch, input_dim, degree-1)
        Cs = tf.transpose(carries[0], perm=[1, 2, 0])
        basis = tf.concat([tf.expand_dims(C0, -1), tf.expand_dims(C1, -1), Cs], axis=-1)
        if self.orthonormal:
            n = tf.range(0, self.degree + 1, dtype=basis.dtype)
            alpha = tf.cast(alpha, basis.dtype)
            log_norm = (
                tf.math.log(np.pi)
                + (1.0 - 2.0 * alpha) * tf.math.log(2.0)
                + tf.math.lgamma(n + 2.0 * alpha)
                - tf.math.lgamma(n + 1.0)
                - tf.math.log(n + alpha)
                - 2.0 * tf.math.lgamma(alpha)
            )
            scale = tf.exp(-0.5 * log_norm)
            basis = basis * scale
        return tf.reshape(basis, (-1, self.input_dim, self.degree + 1))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha_init": self.alpha_init,
                "alpha_trainable": self.alpha_trainable,
                "orthonormal": self.orthonormal,
            }
        )
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="Hermite")
class Hermite(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using (physicist's) Hermite polynomials.

    Domain is :math:`\mathbb{R}`; values grow approximately like :math:`2^{n}`
    so large degrees or large ``|x|`` can overflow in float32. Set ``normalized=True``
    to use probabilist's Hermite (He) which is better conditioned.

    .. warning::

        Hermite polynomials are defined on :math:`\mathbb{R}` but grow rapidly.
        For inputs with ``|x| > 5``, consider using ``input_clip=(-5, 5)`` or
        preprocessing to avoid overflow, especially at higher degrees.

    The (physicist's) Hermite polynomials are generated by the three-term recurrence relation:

    * :math:`{H_{0}(x) = 1}`
    * :math:`{H_{1}(x) = 2x}`
    * :math:`{H_{n+1}(x) = 2 * x * H_{n}(x) - 2 * n * H_{n-1}(x)}` when n >= 0

    See also: https://en.wikipedia.org/wiki/Hermite_polynomials#Recurrence_relation
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        normalized: bool = False,
        input_clip: tuple[float, float] | None = (-5.0, 5.0),
        **kwargs,
    ):
        """
        Parameters
        ----------
        degree : int
            Maximum polynomial degree.
        units : int
            Output dimensionality.
        normalized : bool, default False
            When True, use probabilist's Hermite polynomials :math:`He_n` (better conditioned).
            When False, use physicist's Hermite :math:`H_n`.
        input_clip : tuple[float, float] | None, default (-5.0, 5.0)
            Interval to clamp inputs. Hermite polynomials grow rapidly; clamping to
            ``[-5, 5]`` prevents overflow in typical float32 usage. Set to ``None``
            to disable clamping (use with caution).
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        if degree > 15 and not normalized:
            warnings.warn(
                f"Physicist's Hermite polynomials with degree={degree} > 15 may overflow in float32. "
                "Consider using normalized=True (probabilist's form) or promote_to_float64=True.",
                UserWarning,
                stacklevel=2,
            )
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)
        self.normalized = normalized

    @kan_fn
    def pseudo_vandermonde(self, x):
        # Probabilist's Hermite has milder growth; branch based on configuration.
        hermite_basis = [tf.ones_like(x)]

        if self.degree > 0:
            hermite_basis.append(x if self.normalized else 2.0 * x)

        for n in range(2, self.degree + 1):
            if self.normalized:
                hermite_basis.append((x * hermite_basis[n - 1]) - ((n - 1) * hermite_basis[n - 2]))
            else:
                hermite_basis.append((2.0 * x * hermite_basis[n - 1]) - (2.0 * (n - 1) * hermite_basis[n - 2]))

        return tf.stack(hermite_basis, axis=-1)

    def clenshaw_basis(self, x: tf.Tensor) -> tf.Tensor:
        # Physicist's Hermite via scan recurrence
        x = tf.reshape(x, (-1, self.input_dim))
        H0 = tf.ones_like(x)
        if self.degree == 0:
            return tf.expand_dims(H0, -1)
        H1 = x if self.normalized else 2.0 * x
        if self.degree == 1:
            return tf.stack([H0, H1], axis=-1)

        def step(carry, n):
            Hn_1, Hn_2 = carry
            n_f = tf.cast(n, x.dtype)
            if self.normalized:
                Hn = x * Hn_1 - (n_f - 1.0) * Hn_2
            else:
                Hn = 2.0 * x * Hn_1 - 2.0 * (n_f - 1.0) * Hn_2
            return (Hn, Hn_1)  # Only return new carry

        ns = tf.range(2, self.degree + 1)
        carries = tf.scan(step, ns, initializer=(H1, H0))
        # carries[0] has shape (degree-1, batch, input_dim) - transpose to (batch, input_dim, degree-1)
        Hs = tf.transpose(carries[0], perm=[1, 2, 0])
        basis = tf.concat([tf.expand_dims(H0, -1), tf.expand_dims(H1, -1), Hs], axis=-1)
        return tf.reshape(basis, (-1, self.input_dim, self.degree + 1))

    def get_config(self):
        config = super().get_config()
        config.update({"normalized": self.normalized})
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="Jacobi")
class Jacobi(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Jacobi polynomials.

    The Jacobi polynomials are generated by the three-term recurrence relation:
    Valid for ``alpha > -1`` and ``beta > -1`` with inputs in ``[-1, 1]`` (inputs are clipped by default).

    * :math:`J^{\alpha, \beta}_{0}(x) = 1`
    * :math:`J^{\alpha, \beta}_{1}(x) = \frac{1}{2} * (\alpha + \beta + 2) * x + \frac{1}{2} * (\alpha - \beta)`
    * :math:`J^{\alpha, \beta}_{n+1}(x) = (A^{\alpha, \beta}_{n} * x - B^{\alpha, \beta}_{n}) * J^{\alpha, \beta}_{n}(x) - C^{\alpha, \beta}_{n} * J^{\alpha, \beta}_{n-1}(x)` when n >= 1

    with

    * :math:`A^{\alpha, \beta}_{n} = \frac{(2n + \alpha + \beta +1) * (2n + \alpha + \beta + 2)}{2(n+1) * (n + \alpha + \beta + 1)}`
    * :math:`B^{\alpha, \beta}_{n} = \frac{(\beta^{2} - \alpha^{2})(2n + \alpha + \beta +1)}{2(n+1) * (n + \alpha + \beta + 1)(2n + \alpha + \beta)}`
    * :math:`C^{\alpha, \beta}_{n} = \frac{(n + \alpha)(n + \beta)(2n + \alpha + \beta + 2)}{(n+1) * (n + \alpha + \beta + 1)(2n + \alpha + \beta)}`

    Special cases of the Jacobi polynomials are:
    * the Legendre polynomials (when alpha=beta=0);
    * the Chebyshev polynomials of the first kind (when alpha=beta=-1/2);
    * the Chebyshev polynomials of the second kind (when alpha=beta==1/2);
    * the Gegenbauer polynomials (when alpha=beta)


    TODO:     Jacobi polynomials in hypergeometric representation
    :math:`J^{\alpha, \beta}_{n}(x) = \binom{n + \alpha}{n} * _2F_1(-n, n + \alpha + \beta + 1; \alpha + 1; \frac{1-x}{2})`
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        alpha_init: float | None = None,
        alpha_trainable=True,
        beta_init: float | None = None,
        beta_trainable=True,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        orthonormal: bool = False,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        degree : int
            Maximum polynomial degree.
        units : int
            Output dimensionality.
        alpha_init, beta_init : float | None
            Initial parameters (must satisfy alpha, beta > -1); default RandomNormal when None.
        alpha_trainable, beta_trainable : bool
            Trainability flags.
        input_clip : tuple[float, float] | None
            Optional input clamp (defaults to [-1, 1]).
        orthonormal : bool, default False
            When True, rescales :math:`P_n^{(\\alpha,\\beta)}` to be orthonormal with respect to
            weight :math:`(1-x)^{\\alpha}(1+x)^{\\beta}` on ``[-1,1]``.
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

        self.alpha_init = alpha_init
        self.alpha_trainable = alpha_trainable
        self.beta_init = beta_init
        self.beta_trainable = beta_trainable
        self.orthonormal = orthonormal

        self.alpha_logits = None
        self.beta_logits = None

    def build(self, input_shape):
        super().build(input_shape)

        self.alpha_logits = create_bounded_param_logits(
            self, "alpha", self.alpha_init, lower_bound=-1.0, trainable=self.alpha_trainable
        )
        self.beta_logits = create_bounded_param_logits(
            self, "beta", self.beta_init, lower_bound=-1.0, trainable=self.beta_trainable
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        # softplus_lower_bound ensures alpha, beta > -1 with smooth gradients
        alpha = softplus_lower_bound(tf.cast(self.alpha_logits, tf.float64), lower_bound=-1.0)
        beta = softplus_lower_bound(tf.cast(self.beta_logits, tf.float64), lower_bound=-1.0)

        # See: http://lsec.cc.ac.cn/~hyu/teaching/shonm2013/STWchap3.2p.pdf (section 3.2.1.3)
        # :math:`J^{\alpha, \beta}_{0}(x) = 1`
        jacobi_basis = [tf.ones_like(x, dtype=tf.float64)]

        if self.degree > 0:
            # :math:`J^{\alpha, \beta}_{1}(x) = \frac{1}{2} * (\alpha + \beta + 2) * x + \frac{1}{2} * (\alpha - \beta)`
            jacobi_basis.append(0.5 * (alpha - beta) + (alpha + beta + 2) * x / 2)

        # DLMF 18.9.5: for n >= 1,
        # P_{n+1} = (a_n x + b_n) P_n - c_n P_{n-1}
        for n in range(1, self.degree):
            n_f = tf.cast(n, x.dtype)
            A_n = ((2 * n_f + alpha + beta + 1) * (2 * n_f + alpha + beta + 2)) / (
                2 * (n_f + 1) * (n_f + alpha + beta + 1)
            )
            B_n = (alpha**2 - beta**2) / ((2 * n_f + alpha + beta) * (2 * n_f + alpha + beta + 2))
            C_n = ((n_f + alpha) * (n_f + beta) * (2 * n_f + alpha + beta + 2)) / (
                (n_f + 1) * (n_f + alpha + beta + 1) * (2 * n_f + alpha + beta)
            )

            jacobi_basis.append(((A_n * x) + B_n) * jacobi_basis[n] - C_n * jacobi_basis[n - 1])

        basis = tf.stack(jacobi_basis, axis=-1)
        if self.orthonormal:
            # Norm: 2^{α+β+1} Γ(n+α+1) Γ(n+β+1) / [(2n+α+β+1) n! Γ(n+α+β+1)]
            n = tf.range(0, self.degree + 1, dtype=tf.float64)
            log2 = tf.math.log(tf.constant(2.0, dtype=tf.float64))
            log_norm = (
                (alpha + beta + 1.0) * log2
                + tf.math.lgamma(n + alpha + 1.0)
                + tf.math.lgamma(n + beta + 1.0)
                - tf.math.log(2.0 * n + alpha + beta + 1.0)
                - tf.math.lgamma(n + 1.0)
                - tf.math.lgamma(n + alpha + beta + 1.0)
            )
            scale = tf.exp(-0.5 * log_norm)
            basis = basis * scale

        return tf.cast(basis, orig_dtype)

    def clenshaw_basis(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.cast(tf.reshape(x, (-1, self.input_dim)), tf.float64)
        # softplus_lower_bound ensures alpha, beta > -1 with smooth gradients
        alpha = softplus_lower_bound(tf.cast(self.alpha_logits, tf.float64), lower_bound=-1.0)
        beta = softplus_lower_bound(tf.cast(self.beta_logits, tf.float64), lower_bound=-1.0)
        orig_dtype = x.dtype

        P0 = tf.ones_like(x, dtype=tf.float64)
        if self.degree == 0:
            return tf.cast(tf.expand_dims(P0, -1), orig_dtype)
        P1 = 0.5 * (alpha - beta) + 0.5 * (alpha + beta + 2.0) * x
        if self.degree == 1:
            return tf.cast(tf.stack([P0, P1], axis=-1), orig_dtype)

        def step(carry, n):
            Pn_1, Pn_2 = carry
            n_f = tf.cast(n, x.dtype)
            # Match pseudo_vandermonde: for n=2,3,... computing P_n from P_{n-1}, P_{n-2}
            # Use (n-1) for coefficient indices since pseudo_vandermonde uses n in range(1, degree)
            n_coef = n_f - 1.0
            A_n = ((2.0 * n_coef + alpha + beta + 1.0) * (2.0 * n_coef + alpha + beta + 2.0)) / (
                2.0 * (n_coef + 1.0) * (n_coef + alpha + beta + 1.0)
            )
            B_n = (alpha**2 - beta**2) / ((2.0 * n_coef + alpha + beta) * (2.0 * n_coef + alpha + beta + 2.0))
            C_n = ((n_coef + alpha) * (n_coef + beta) * (2.0 * n_coef + alpha + beta + 2.0)) / (
                (n_coef + 1.0) * (n_coef + alpha + beta + 1.0) * (2.0 * n_coef + alpha + beta)
            )
            Pn = ((A_n * x) + B_n) * Pn_1 - C_n * Pn_2
            return (Pn, Pn_1)  # Only return new carry

        ns = tf.range(2, self.degree + 1)
        carries = tf.scan(step, ns, initializer=(P1, P0))
        # carries[0] has shape (degree-1, batch, input_dim) - the Pn values
        # Transpose to (batch, input_dim, degree-1) and concat with P0, P1
        Ps = tf.transpose(carries[0], perm=[1, 2, 0])
        basis = tf.concat([tf.expand_dims(P0, -1), tf.expand_dims(P1, -1), Ps], axis=-1)
        if self.orthonormal:
            n = tf.range(0, self.degree + 1, dtype=tf.float64)
            log2 = tf.math.log(tf.constant(2.0, dtype=tf.float64))
            log_norm = (
                (alpha + beta + 1.0) * log2
                + tf.math.lgamma(n + alpha + 1.0)
                + tf.math.lgamma(n + beta + 1.0)
                - tf.math.log(2.0 * n + alpha + beta + 1.0)
                - tf.math.lgamma(n + 1.0)
                - tf.math.lgamma(n + alpha + beta + 1.0)
            )
            scale = tf.exp(-0.5 * log_norm)
            basis = basis * scale
        return tf.cast(tf.reshape(basis, (-1, self.input_dim, self.degree + 1)), orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha_init": self.alpha_init,
                "alpha_trainable": self.alpha_trainable,
                "beta_init": self.beta_init,
                "beta_trainable": self.beta_trainable,
                "orthonormal": self.orthonormal,
            }
        )
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="GeneralizedLaguerre")
class GeneralizedLaguerre(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Laguerre polynomials.

    The generalized Laguerre polynomials are generated by the three-term recurrence relation:
    The orthogonality interval is :math:`[0, \infty)` with weight :math:`x^{\alpha} e^{-x}`; ``alpha`` must satisfy ``alpha > -1``.

    * :math:`L^{\alpha}_{0}(x) = 1`
    * :math:`L^{\alpha}_{1}(x) = 1 + \alpha - x`
    * :math:`L^{\alpha}_{n+1}(x) = \frac{(2n + 1 + \alpha - x) * L^{\alpha}_{n}(x) - (n + \alpha) * L^{\alpha}_{n-1}(x)}{n+1}` when n >= 1

    Special cases of the  generalized Laguerre polynomials are:
    * the Laguerre polynomials (when alpha=0);

    See also: https://en.wikipedia.org/wiki/Laguerre_polynomials#Generalized_Laguerre_polynomials
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        alpha_init: float | None = None,
        alpha_trainable: bool = True,
        normalized: bool = False,
        input_clip: tuple[float, float] | None = (0.0, float('inf')),
        **kwargs,
    ):
        """
        Parameters
        ----------
        degree : int
            Maximum polynomial degree.
        units : int
            Output dimensionality.
        alpha_init : float | None
            Initial ``alpha``; must satisfy ``alpha > -1``; defaults to RandomNormal when None.
        alpha_trainable : bool
            Whether ``alpha`` is trainable.
        normalized : bool, default False
            When True, use orthonormal Laguerre polynomials (rescaled by the square root
            of the normalization constant). The standard form has norm
            :math:`\Gamma(n + \\alpha + 1) / n!`.
        input_clip : tuple[float, float] | None
            Input clamp; defaults to ``(0.0, inf)`` for the natural domain :math:`[0, \\infty)`.
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

        self.alpha_init = alpha_init
        self.alpha_trainable = alpha_trainable
        self.normalized = normalized
        self.alpha_logits = None

        # Validate alpha_init at construction time if explicitly provided
        if alpha_init is not None and alpha_init <= -1.0:
            raise ValueError(
                f"GeneralizedLaguerre requires alpha > -1, got alpha_init={alpha_init}. "
                "The weight function x^alpha * exp(-x) is only integrable on [0, inf) for alpha > -1."
            )

    def build(self, input_shape):
        super().build(input_shape)
        self.alpha_logits = create_bounded_param_logits(
            self, "alpha", self.alpha_init, lower_bound=-1.0, trainable=self.alpha_trainable
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        # softplus_lower_bound ensures alpha > -1 with smooth gradients
        alpha = softplus_lower_bound(tf.cast(self.alpha_logits, x.dtype), lower_bound=-1.0)
        # :math:`L^{\alpha}_{0}(x) = 1`
        laguerre_basis = [tf.ones_like(x)]

        if self.degree > 0:
            # :math:`L^{\alpha}_{1}(x) = 1 + \alpha - x`
            laguerre_basis.append(1.0 + alpha - x)

        for n in range(1, self.degree):
            # :math:`(n+1) L_{n+1}^{\\alpha}(x) = (2n + 1 + \\alpha - x) L_n^{\\alpha}(x) - (n + \\alpha) L_{n-1}^{\\alpha}(x)`
            n_f = tf.cast(n, x.dtype)
            num = (2 * n_f + 1 + alpha - x) * laguerre_basis[n] - (n_f + alpha) * laguerre_basis[n - 1]
            laguerre_basis.append(num / (n_f + 1.0))

        basis = tf.stack(laguerre_basis, axis=-1)
        if self.normalized:
            # Norm: Γ(n + α + 1) / n!
            n = tf.range(0, self.degree + 1, dtype=basis.dtype)
            alpha_cast = tf.cast(alpha, basis.dtype)
            log_norm = tf.math.lgamma(n + alpha_cast + 1.0) - tf.math.lgamma(n + 1.0)
            scale = tf.exp(-0.5 * log_norm)
            basis = basis * scale
        return basis

    def clenshaw_basis(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute Laguerre basis using tf.scan for memory efficiency at high degrees.

        Uses the recurrence:
        :math:`L^{\\alpha}_{n+1}(x) = \\frac{(2n+1+\\alpha-x) L^{\\alpha}_n(x) - (n+\\alpha) L^{\\alpha}_{n-1}(x)}{n+1}`
        """
        x = tf.reshape(x, (-1, self.input_dim))
        # softplus_lower_bound ensures alpha > -1 with smooth gradients
        alpha = softplus_lower_bound(tf.cast(self.alpha_logits, x.dtype), lower_bound=-1.0)

        L0 = tf.ones_like(x)
        if self.degree == 0:
            return tf.expand_dims(L0, -1)
        L1 = 1.0 + alpha - x
        if self.degree == 1:
            return tf.stack([L0, L1], axis=-1)

        def step(carry, n):
            Ln_1, Ln_2 = carry
            n_f = tf.cast(n, x.dtype)
            # L_{n+1} = ((2n+1+alpha-x) L_n - (n+alpha) L_{n-1}) / (n+1)
            Ln = ((2.0 * n_f + 1.0 + alpha - x) * Ln_1 - (n_f + alpha) * Ln_2) / (n_f + 1.0)
            return (Ln, Ln_1)  # Only return new carry

        ns = tf.range(1, self.degree)  # n=1 gives L2, n=degree-1 gives L_degree
        carries = tf.scan(step, ns, initializer=(L1, L0))
        # carries[0] has shape (degree-1, batch, input_dim) - transpose to (batch, input_dim, degree-1)
        Ls = tf.transpose(carries[0], perm=[1, 2, 0])
        basis = tf.concat([tf.expand_dims(L0, -1), tf.expand_dims(L1, -1), Ls], axis=-1)
        if self.normalized:
            n = tf.range(0, self.degree + 1, dtype=basis.dtype)
            alpha_cast = tf.cast(alpha, basis.dtype)
            log_norm = tf.math.lgamma(n + alpha_cast + 1.0) - tf.math.lgamma(n + 1.0)
            scale = tf.exp(-0.5 * log_norm)
            basis = basis * scale
        return tf.reshape(basis, (-1, self.input_dim, self.degree + 1))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha_init": self.alpha_init,
                "alpha_trainable": self.alpha_trainable,
                "normalized": self.normalized,
            }
        )
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="Legendre")
class Legendre(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Legendre polynomials.

    Legendre polynomials :math:`P_n(x)` are orthogonal on :math:`[-1, 1]` with unit weight:

    .. math::

        \int_{-1}^{1} P_m(x) \, P_n(x) \, dx = \frac{2}{2n+1} \delta_{mn}

    They satisfy the three-term recurrence:

    .. math::

        P_0(x) &= 1 \\
        P_1(x) &= x \\
        (n+1) P_{n+1}(x) &= (2n+1) \, x \, P_n(x) - n \, P_{n-1}(x)

    When ``orthonormal=True``, the basis is rescaled to:

    .. math::

        \tilde{P}_n(x) = \sqrt{\frac{2n+1}{2}} \, P_n(x)

    so that :math:`\int_{-1}^{1} \tilde{P}_m(x) \tilde{P}_n(x) dx = \delta_{mn}`.

    Notes
    -----
    Legendre polynomials are excellent for general function approximation on bounded
    intervals and have near-minimax properties. They are related to Chebyshev polynomials
    and share similar convergence characteristics for smooth functions.

    See Also
    --------
    Chebyshev1st : Often preferred for numerical stability
    Jacobi : Generalization with parameters :math:`\\alpha, \\beta`
    Gegenbauer : Special case of Jacobi with :math:`\\alpha = \\beta`

    References
    ----------
    .. [DLMF] NIST Digital Library of Mathematical Functions, Chapter 18
       https://dlmf.nist.gov/18
    .. [Wikipedia] https://en.wikipedia.org/wiki/Legendre_polynomials
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        orthonormal: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        degree : int
            Maximum polynomial degree.
        units : int | None
            Output dimensionality. If omitted, ``output_dim`` is accepted for backward compatibility.
        output_dim : int | None
            Legacy alias for ``units``.
        input_clip : tuple[float, float] | None
            Interval to clip inputs to the canonical domain [-1, 1].
        orthonormal : bool, default False
            When True, rescales :math:`P_n` to :math:`\\sqrt{\\frac{2n+1}{2}} P_n` (orthonormal on ``[-1,1]``).
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        resolved_units = units if units is not None else output_dim
        super().__init__(degree=degree, units=resolved_units, input_clip=input_clip, **kwargs)
        self.orthonormal = orthonormal

    @kan_fn
    def pseudo_vandermonde(self, x):
        # :math:`P_{0}(x) = 1`
        legendre_basis = [tf.ones_like(x)]

        if self.degree > 0:
            # :math:`P_{1}(x) = x`
            legendre_basis.append(x)

        for n in range(2, self.degree + 1):
            # :math:`P_{n+1}(x) = \frac{(2n + 1) * x * P_{n}(x) - n * P_{n-1}(x)}{n+1}` when n >= 1
            legendre_basis.append((((2 * n - 1) * x * legendre_basis[n - 1]) - ((n - 1) * legendre_basis[n - 2])) / n)

        basis = tf.stack(legendre_basis, axis=-1)
        if self.orthonormal:
            n = tf.range(0, self.degree + 1, dtype=basis.dtype)
            scale = tf.sqrt((2.0 * n + 1.0) / 2.0)
            basis = basis * scale
        return basis

    def clenshaw_basis(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute Legendre basis using tf.scan for memory efficiency at high degrees.

        Uses the recurrence:
        :math:`P_{n+1}(x) = \\frac{(2n+1) x P_n(x) - n P_{n-1}(x)}{n+1}`
        """
        x = tf.reshape(x, (-1, self.input_dim))
        P0 = tf.ones_like(x)
        if self.degree == 0:
            basis = tf.expand_dims(P0, -1)
            if self.orthonormal:
                basis = basis * tf.sqrt(0.5)
            return basis
        P1 = x
        if self.degree == 1:
            basis = tf.stack([P0, P1], axis=-1)
            if self.orthonormal:
                n = tf.range(0, 2, dtype=basis.dtype)
                scale = tf.sqrt((2.0 * n + 1.0) / 2.0)
                basis = basis * scale
            return basis

        def step(carry, n):
            Pn_1, Pn_2 = carry
            n_f = tf.cast(n, x.dtype)
            # P_{n+1} = ((2n+1) x P_n - n P_{n-1}) / (n+1)
            Pn = ((2.0 * n_f + 1.0) * x * Pn_1 - n_f * Pn_2) / (n_f + 1.0)
            return (Pn, Pn_1)  # Only return new carry

        ns = tf.range(1, self.degree)  # n=1 gives P2, n=degree-1 gives P_degree
        carries = tf.scan(step, ns, initializer=(P1, P0))
        # carries[0] has shape (degree-1, batch, input_dim) - the Pn values
        # Transpose to (batch, input_dim, degree-1) and concat with P0, P1
        Ps = tf.transpose(carries[0], perm=[1, 2, 0])
        basis = tf.concat([tf.expand_dims(P0, -1), tf.expand_dims(P1, -1), Ps], axis=-1)
        if self.orthonormal:
            n = tf.range(0, self.degree + 1, dtype=basis.dtype)
            scale = tf.sqrt((2.0 * n + 1.0) / 2.0)
            basis = basis * scale
        return tf.reshape(basis, (-1, self.input_dim, self.degree + 1))

    def true_clenshaw_eval(self, x: tf.Tensor, coefficients: tf.Tensor) -> tf.Tensor:
        """
        True Clenshaw summation for Legendre polynomials with O(1) memory.

        Fuses basis computation with coefficient contraction using the Clenshaw
        recurrence for Legendre polynomials:

        .. math::

            b_{n+1} = 0, \\quad b_n = c_n \\\\
            b_{k-1} = c_{k-1} + \\frac{2k+1}{k+1} x \\cdot b_k - \\frac{k+1}{k+2} b_{k+1} \\\\
            P_0 c_0 + P_1 c_1 + \\cdots + P_n c_n = b_0

        This avoids materializing the full basis tensor, critical for high-degree
        polynomials on memory-constrained accelerators.

        :param x: Input tensor of shape (batch, input_dim).
        :param coefficients: Coefficient tensor of shape (input_dim, degree+1, output_dim).
        :returns: Output tensor of shape (batch, output_dim).
        """
        # Reshape x for computation: (batch, input_dim)
        x = tf.reshape(x, (-1, self.input_dim))  # (B, D)
        
        # coefficients: (input_dim, degree+1, output_dim)
        degree = self.degree
        
        # Apply orthonormal scaling to coefficients if needed
        if self.orthonormal:
            n = tf.range(0, degree + 1, dtype=coefficients.dtype)
            scale = tf.sqrt((2.0 * n + 1.0) / 2.0)
            # scale: (degree+1,) -> (1, degree+1, 1) for broadcasting
            coefficients = coefficients * tf.reshape(scale, (1, -1, 1))
        
        # Legendre Clenshaw recurrence (backward):
        # b_{k} = c_k + alpha_k * x * b_{k+1} - beta_{k+1} * b_{k+2}
        # where alpha_k = (2k+1)/(k+1), beta_{k+1} = (k+1)/(k+2)
        
        # Initialize: b_{n+1} = 0, b_n = c_n
        batch_size = tf.shape(x)[0]
        output_dim = tf.shape(coefficients)[-1]
        
        b_next2 = tf.zeros((batch_size, self.input_dim, output_dim), dtype=coefficients.dtype)
        b_next1 = tf.broadcast_to(
            tf.expand_dims(coefficients[:, degree, :], 0),  # (1, D, O)
            (batch_size, self.input_dim, output_dim)
        )
        
        # x for broadcasting: (B, D) -> (B, D, 1)
        x_expanded = tf.expand_dims(x, -1)
        
        def clenshaw_step(carry, k):
            b_next1, b_next2 = carry
            k_f = tf.cast(k, coefficients.dtype)
            
            # Coefficients for Legendre recurrence
            alpha = (2.0 * k_f + 1.0) / (k_f + 1.0)  # (2k+1)/(k+1)
            beta = (k_f + 1.0) / (k_f + 2.0)  # (k+1)/(k+2)
            
            # c_k coefficient
            c_k = coefficients[:, k, :]  # (D, O)
            c_k = tf.broadcast_to(tf.expand_dims(c_k, 0), tf.shape(b_next1))
            
            # b_k = c_k + alpha_k * x * b_{k+1} - beta_{k+1} * b_{k+2}
            b_new = c_k + alpha * x_expanded * b_next1 - beta * b_next2
            
            return (b_new, b_next1)
        
        # Run recurrence from k = degree-1 down to k = 0
        if degree > 0:
            ks = tf.range(degree - 1, -1, -1)  # degree-1, degree-2, ..., 0
            (b_0, _) = tf.foldl(clenshaw_step, ks, initializer=(b_next1, b_next2))
        else:
            b_0 = b_next1
        
        # For Legendre, the result is simply b_0 (P_0(x) = 1)
        # Sum over input dimensions: (B, D, O) -> (B, O)
        output = tf.reduce_sum(b_0, axis=1)
        
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"orthonormal": self.orthonormal})
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="AssociatedMeixnerPollaczek")
class AssociatedMeixnerPollaczek(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Associated Meixner–Pollaczek polynomials.

    Meixner-Pollaczek polynomials are orthogonal on the real line with respect to the weight function given by the Meixner-Pollaczek distribution.
    The Associated Meixner–Pollaczek polynomials are generated by the three-term recurrence relation:

    * :math:`P^{\lambda}_{-1}(x; \phi, c) = 0`
    * :math:`P^{\lambda}_{0}(x; \phi, c)  = 1`
    * :math:`P^{\lambda}_{n+1}(x; \phi, c) = \frac{(2 * x * \sin(\phi) + 2*(n + c + \lambda)* P^{\lambda}_{n}(x; \phi, c) - (n + c + 2*\lambda - 1) * P^{\lambda}_{n-1}(x; \phi, c)}{n + c + 1}, \, n >= 0`

    See also: https://dlmf.nist.gov/18.30#v
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        lambda_init: float | None = None,
        lambda_trainable=True,
        phi_init: float | None = None,
        phi_trainable=True,
        c_init: float | None = None,
        c_trainable=True,
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
        lambda_init, phi_init, c_init : float | None
            Initial parameter values; default to RandomNormal when None.
        lambda_trainable, phi_trainable, c_trainable : bool
            Trainability flags.
        input_clip : tuple[float, float] | None
            Optional input clamp.
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

        self.lambda_init = lambda_init
        self.lambda_trainable = lambda_trainable
        self.phi_init = phi_init
        self.phi_trainable = phi_trainable
        self.c_init = c_init
        self.c_trainable = c_trainable

        self.lambda_ = None
        self.phi = None
        self.c = None

    def build(self, input_shape):
        super().build(input_shape)

        self.lambda_ = create_trainable_param(self, "lambda", self.lambda_init, trainable=self.lambda_trainable)
        self.phi = create_trainable_param(self, "phi", self.phi_init, trainable=self.phi_trainable)
        self.c = create_trainable_param(self, "c", self.c_init, trainable=self.c_trainable)

    @kan_fn
    def pseudo_vandermonde(self, x):
        # Cast trainable params to x.dtype to avoid dtype mismatch
        lambda_ = tf.cast(self.lambda_, x.dtype)
        phi = tf.cast(self.phi, x.dtype)
        c = tf.cast(self.c, x.dtype)

        # :math:`P^{\lambda}_{0}(x; \phi, c)  = 1`
        meixner_pollaczek_basis = [tf.ones_like(x)]

        if self.degree > 0:
            # \frac{(2 * x * \sin(\phi) + 2*(n + c + \lambda)}{n + c + 1}
            meixner_pollaczek_basis.append(
                (2 * x * tf.sin(phi) + 2 * (1 + c + lambda_) * tf.cos(phi)) / (1 + c + 1.0)
            )

        for n in range(2, self.degree + 1):
            # :math:`P^{\lambda}_{n+1}(x; \phi, c) = \frac{(2 * x * \sin(\phi) + 2*(n + c + \lambda)* P^{\lambda}_{n}(x; \phi, c) - (n + c + 2*\lambda - 1) * P^{\lambda}_{n-1}(x; \phi, c)}{n + c + 1}  when n >= 2
            term1 = 2 * x * tf.sin(phi) + 2 * (n + c + lambda_) * tf.cos(phi)
            term2 = n + c + 2 * lambda_ - 1.0
            term3 = n + c + 1.0
            meixner_pollaczek_basis.append(
                (term1 * meixner_pollaczek_basis[n - 1] - term2 * meixner_pollaczek_basis[n - 2]) / term3
            )

        return tf.stack(meixner_pollaczek_basis, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lambda_init": self.lambda_init,
                "lambda_trainable": self.lambda_trainable,
                "phi_init": self.phi_init,
                "phi_trainable": self.phi_trainable,
                "c_init": self.c_init,
                "c_trainable": self.c_trainable,
            }
        )
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="Pollaczek")
class Pollaczek(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Pollaczek polynomials.

    The Pollaczek polynomials are generated by the three-term recurrence relation:

    * :math:`P_{0}(x; a, b) = 1`
    * :math:`P_{1}(x; a, b) = (2 * a + 1) * x + 2 * b`
    * :math:`P_{n}(x;a,b) = \frac{[(2n-1+2a)x+2b]P_{n-1}(x;a,b)-(n-1)P_{n-2}(x;a,b) }{n}, \, n \geq 2`

    See also: https://mathworld.wolfram.com/PollaczekPolynomial.html
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
        a_init, b_init : float | None
            Initial parameter values; default to RandomNormal when None.
        a_trainable, b_trainable : bool
            Trainability flags.
        input_clip : tuple[float, float] | None
            Optional input clamp.
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

        self.a_init = a_init
        self.a_trainable = a_trainable
        self.b_init = b_init
        self.b_trainable = b_trainable
        self.a = None
        self.b = None

    def build(self, input_shape):
        super().build(input_shape)

        def _init_param(v):
            if v is not None:
                return tfk.initializers.Constant(value=v)
            return tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)

        self.a = self.add_weight(initializer=_init_param(self.a_init), name="a", trainable=self.a_trainable)

        self.b = self.add_weight(initializer=_init_param(self.b_init), name="b", trainable=self.b_trainable)

    @kan_fn
    def pseudo_vandermonde(self, x):
        # Cast trainable params to x.dtype to avoid dtype mismatch
        a = tf.cast(self.a, x.dtype)
        b = tf.cast(self.b, x.dtype)

        # :math:`P_{0}(x; a, b) = 1`
        pollaczek_basis = [tf.ones_like(x)]

        if self.degree > 0:
            # :math:`P_{1}(x; a, b) = (2 * a + 1) * x + 2 * b`
            pollaczek_basis.append((2 * a + 1) * x + 2 * b)

        for n in range(2, self.degree + 1):
            # :math:`P_{n}(x;a,b) = \frac{[(2n-1+2a)x+2b]P_{n-1}(x;a,b)-(n-1)P_{n-2}(x;a,b) }{n} when n >= 2
            pollaczek_basis.append(
                (
                    ((2 * n - 1 + 2 * a) * x + 2 * b) * pollaczek_basis[n - 1]
                    - (n - 1) * pollaczek_basis[n - 2]
                )
                / n
            )

        return tf.stack(pollaczek_basis, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "a_init": self.a_init,
                "a_trainable": self.a_trainable,
                "b_init": self.b_init,
                "b_trainable": self.b_trainable,
            }
        )
        return config


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
            import math
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
