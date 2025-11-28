## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""
q-Orthogonal Polynomial KAN layers (q-Hahn Class and Askey-Wilson Class).

This module provides KAN layers using q-analog polynomial bases:

**Sprint 7E - q-Hahn Class (Part 1):**
- QHahn: q-analog of Hahn polynomials
- BigQJacobi: Big q-Jacobi polynomials  
- LittleQJacobi: Little q-Jacobi polynomials
- QMeixner: q-analog of Meixner polynomials
- QKrawtchouk: q-analog of Krawtchouk polynomials

**Sprint 7F - q-Polynomials (Part 2):**
- QCharlier: q-analog of Charlier polynomials
- QRacah: q-Racah polynomials (Askey-Wilson class)
- DualQHahn: Dual q-Hahn polynomials
- DualQKrawtchouk: Dual q-Krawtchouk polynomials
- AffineQKrawtchouk: Affine q-Krawtchouk polynomials

All q-polynomials satisfy the q-difference equation and use three-term recurrence
for numerical stability. The parameter q is constrained to (0, 1) for convergence.

References
----------
.. [1] NIST DLMF 18.27 - q-Hahn Class
.. [2] NIST DLMF 18.28 - Askey-Wilson Class
.. [3] Koekoek, R., Lesky, P. A., & Swarttouw, R. F. (2010). 
       Hypergeometric orthogonal polynomials and their q-analogues.
"""
import math
from abc import ABC, abstractmethod

import tensorflow as tf

from arnold.utils.compilation import kan_function
from arnold.utils.constraints import softplus_lower_bound

from .poly_base import PolynomialBase


tfk = tf.keras
kan_fn = kan_function()


def _inverse_softplus_lower_bound(value: float, lower_bound: float, eps: float = 1e-6) -> float:
    """
    Compute logits that produce `value` when passed through softplus_lower_bound.
    
    Python-only version for graph-mode compatibility.
    """
    y = value - lower_bound - eps
    if y <= 0:
        raise ValueError(f"value={value} must be > lower_bound + eps = {lower_bound + eps}")
    if y > 20:
        return y
    return math.log(math.exp(y) - 1)


def _inverse_sigmoid(value: float, eps: float = 1e-6) -> float:
    """
    Compute logit that produces `value` when passed through sigmoid.
    
    Clamps value to (eps, 1-eps) for numerical stability.
    """
    value = max(eps, min(1 - eps, value))
    return math.log(value / (1 - value))


class QPolynomialBase(PolynomialBase, ABC):
    r"""
    Abstract base class for q-orthogonal polynomial KAN layers.

    All q-polynomials in the q-Hahn class share a common parameter q ∈ (0, 1).
    This base class provides:
    - q parameter management with sigmoid constraint
    - Common q-Pochhammer symbol computation
    - Common q-power precomputation utilities

    The q-Pochhammer symbol is defined as:

    .. math::

        (a; q)_n = \prod_{k=0}^{n-1} (1 - a q^k)

    with :math:`(a; q)_0 = 1`.

    Parameters
    ----------
    q : float
        Base parameter, must be in (0, 1). Default is 0.5.
    q_trainable : bool
        Whether q is trainable. Default is True.
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)
        
        if not 0 < q < 1:
            raise ValueError(f"q must be in (0, 1), got {q}")
        
        self.q_init = q
        self.q_trainable = q_trainable
        self.q_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        
        # q ∈ (0, 1) via sigmoid
        q_logit = _inverse_sigmoid(self.q_init)
        
        self.q_logits = self.add_weight(
            name="q_logits",
            shape=(),
            initializer=tf.constant_initializer(q_logit),
            trainable=self.q_trainable,
            dtype=self.dtype,
        )

    def _get_q(self, dtype=tf.float64):
        """Get q parameter constrained to (0, 1)."""
        return tf.sigmoid(tf.cast(self.q_logits, dtype))

    def _q_pochhammer(self, a, q, n, dtype=tf.float64):
        """
        Compute q-Pochhammer symbol (a; q)_n.
        
        (a; q)_n = prod_{k=0}^{n-1} (1 - a * q^k)
        
        Returns scalar tensor.
        """
        if n == 0:
            return tf.ones((), dtype=dtype)
        
        # Compute product: (1 - a) * (1 - a*q) * ... * (1 - a*q^{n-1})
        k = tf.range(n, dtype=dtype)
        factors = 1.0 - a * tf.pow(q, k)
        return tf.reduce_prod(factors)

    def _q_pochhammer_ratio(self, a, q, n, m, dtype=tf.float64):
        """
        Compute ratio (a; q)_n / (a; q)_m for n >= m.
        
        More numerically stable than computing separately.
        """
        if n == m:
            return tf.ones((), dtype=dtype)
        if n < m:
            # Reciprocal
            return 1.0 / self._q_pochhammer_ratio(a, q, m, n, dtype)
        
        # (a; q)_n / (a; q)_m = prod_{k=m}^{n-1} (1 - a * q^k)
        k = tf.range(m, n, dtype=dtype)
        factors = 1.0 - a * tf.pow(q, k)
        return tf.reduce_prod(factors)

    def get_config(self):
        config = super().get_config()
        config.update({
            "q": self.q_init,
            "q_trainable": self.q_trainable,
        })
        return config

    @abstractmethod
    def pseudo_vandermonde(self, x):
        """Compute q-polynomial basis; implemented by subclasses."""
        raise NotImplementedError


@tfk.utils.register_keras_serializable(package="arnold", name="QHahn")
class QHahn(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using q-Hahn polynomials.

    The q-Hahn polynomials :math:`Q_n(x; \alpha, \beta, N; q)` are defined as:

    .. math::

        Q_n(x) = Q_n(x; \alpha, \beta, N; q) = 
            {}_3\phi_2\left(\begin{array}{c} q^{-n}, \alpha\beta q^{n+1}, x \\
            \alpha q, q^{-N} \end{array}; q, q\right)

    for :math:`n = 0, 1, \ldots, N`.

    The orthogonality relation is:

    .. math::

        \sum_{y=0}^{N} Q_n(q^{-y}) Q_m(q^{-y}) \binom{N}{y}_q 
            \frac{(\alpha q; q)_y (\beta q; q)_{N-y}}{(\alpha\beta q^2; q)_N} 
            (\alpha q)^y = h_n \delta_{n,m}

    Three-term recurrence:

    .. math::

        x Q_n(x) = A_n Q_{n+1}(x) + B_n Q_n(x) + C_n Q_{n-1}(x)

    Parameters
    ----------
    alpha : float
        Parameter α > 0 or α < -q^{-1}.
    beta : float
        Parameter β > 0 or β < -q^{-1}.
    N : int
        Discrete support size, must be >= degree.
    q : float
        Base parameter in (0, 1).

    Notes
    -----
    In the limit q → 1, q-Hahn polynomials reduce to classical Hahn polynomials.

    References
    ----------
    .. [1] NIST DLMF 18.27(ii) - q-Hahn Polynomials
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        N: int = 10,
        alpha_trainable: bool = False,
        beta_trainable: bool = False,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        if N < degree:
            raise ValueError(f"N must be >= degree, got N={N}, degree={degree}")
        
        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )
        
        self.alpha_init = alpha
        self.beta_init = beta
        self.N = N
        self.alpha_trainable = alpha_trainable
        self.beta_trainable = beta_trainable
        
        self.alpha_logits = None
        self.beta_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        
        # α, β > 0 via softplus
        alpha_logit = _inverse_softplus_lower_bound(self.alpha_init, 0.0)
        beta_logit = _inverse_softplus_lower_bound(self.beta_init, 0.0)
        
        self.alpha_logits = self.add_weight(
            name="alpha_logits",
            shape=(),
            initializer=tf.constant_initializer(alpha_logit),
            trainable=self.alpha_trainable,
            dtype=self.dtype,
        )
        self.beta_logits = self.add_weight(
            name="beta_logits",
            shape=(),
            initializer=tf.constant_initializer(beta_logit),
            trainable=self.beta_trainable,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        """Compute q-Hahn polynomial basis using three-term recurrence."""
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        
        q = self._get_q(tf.float64)
        alpha = softplus_lower_bound(tf.cast(self.alpha_logits, tf.float64), 0.0)
        beta = softplus_lower_bound(tf.cast(self.beta_logits, tf.float64), 0.0)
        N = tf.cast(self.N, tf.float64)
        
        ones = tf.ones_like(x, dtype=tf.float64)
        
        # Q_0(x) = 1
        Q0 = ones
        basis = [Q0]
        
        if self.degree >= 1:
            # Three-term recurrence coefficients for q-Hahn
            # From Koekoek et al. (2010), the monic form has:
            # x Q_n = Q_{n+1} + B_n Q_n + C_n Q_{n-1}
            # where the coefficients involve q-Pochhammer symbols
            
            # For n=0: Q_1(x) = (x - B_0) / A_0 where A_0 = 1 for monic
            # B_0 = (1 + alpha*q)(1 - q^{-N}) / (alpha*beta*q^2 - 1) simplified form
            
            # Use explicit recurrence from DLMF:
            # A_n, B_n, C_n coefficients for the three-term relation
            
            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for q-Hahn."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)
                
                # Numerator/denominator terms
                ab_q = alpha * beta * q
                
                # A_n coefficient (for Q_{n+1})
                denom1 = (1.0 - ab_q * tf.pow(q, 2.0 * n_f + 1.0))
                denom2 = (1.0 - ab_q * tf.pow(q, 2.0 * n_f + 2.0))
                
                A_n = (1.0 - q_n * tf.pow(q, 1.0)) * (1.0 - alpha * q_n * q) * \
                      (1.0 - beta * q_n * q) * (1.0 - tf.pow(q, n_f - N))
                A_n = A_n / (denom1 * denom2 + 1e-12)
                
                # C_n coefficient (for Q_{n-1})
                C_n = -alpha * beta * q * q_n * (1.0 - q_n) * \
                      (1.0 - ab_q * tf.pow(q, n_f + N)) * \
                      (1.0 - ab_q * tf.pow(q, n_f))
                denom3 = (1.0 - ab_q * tf.pow(q, 2.0 * n_f))
                C_n = C_n / (denom1 * denom3 + 1e-12)
                
                # B_n = 1 - A_n - C_n for normalized form, but we use explicit formula
                B_n = 1.0 - A_n - C_n
                
                return A_n, B_n, C_n
            
            # Q_1: Use simplified first step
            # Q_1(x) = (x - (1 + alpha*q)) / (1 - q^{-N})  (simplified)
            A0, B0, C0 = compute_coeffs(0)
            Q1 = (x - B0 * ones) / (A0 + 1e-12)
            basis.append(Q1)
            
            # Q_n for n >= 2
            Q_prev = Q0
            Q_curr = Q1
            
            for n in range(1, self.degree):
                A_n, B_n, C_n = compute_coeffs(n)
                # x Q_n = A_n Q_{n+1} + B_n Q_n + C_n Q_{n-1}
                # => Q_{n+1} = (x Q_n - B_n Q_n - C_n Q_{n-1}) / A_n
                Q_next = (x * Q_curr - B_n * Q_curr - C_n * Q_prev) / (A_n + 1e-12)
                basis.append(Q_next)
                Q_prev = Q_curr
                Q_curr = Q_next
        
        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha_init,
            "beta": self.beta_init,
            "N": self.N,
            "alpha_trainable": self.alpha_trainable,
            "beta_trainable": self.beta_trainable,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="BigQJacobi")
class BigQJacobi(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Big q-Jacobi polynomials.

    The Big q-Jacobi polynomials :math:`P_n(x; a, b, c; q)` are defined as:

    .. math::

        P_n(x; a, b, c; q) = {}_3\phi_2\left(\begin{array}{c} 
            q^{-n}, abq^{n+1}, x \\ aq, cq \end{array}; q, q\right)

    They are orthogonal with respect to a discrete measure on 
    :math:`\{aq^{k+1}\}_{k=0}^{\infty} \cup \{cq^{k+1}\}_{k=0}^{\infty}`.

    Three-term recurrence:

    .. math::

        x P_n(x) = A_n P_{n+1}(x) + B_n P_n(x) + C_n P_{n-1}(x)

    In the limit q → 1, Big q-Jacobi polynomials reduce to Jacobi polynomials.

    Parameters
    ----------
    a : float
        Parameter, 0 < a < q^{-1}.
    b : float
        Parameter, 0 < b < q^{-1}.
    c : float
        Parameter, c < 0.
    q : float
        Base parameter in (0, 1).

    References
    ----------
    .. [1] NIST DLMF 18.27(iii) - Big q-Jacobi Polynomials
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        a: float = 0.5,
        b: float = 0.5,
        c: float = -0.5,
        a_trainable: bool = False,
        b_trainable: bool = False,
        c_trainable: bool = False,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )
        
        self.a_init = a
        self.b_init = b
        self.c_init = c
        self.a_trainable = a_trainable
        self.b_trainable = b_trainable
        self.c_trainable = c_trainable
        
        self.a_logits = None
        self.b_logits = None
        self.c_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        
        # a, b > 0 via softplus; c < 0 via negative softplus
        a_logit = _inverse_softplus_lower_bound(self.a_init, 0.0)
        b_logit = _inverse_softplus_lower_bound(self.b_init, 0.0)
        # For c < 0, we store |c| and negate in forward pass
        c_logit = _inverse_softplus_lower_bound(abs(self.c_init), 0.0)
        
        self.a_logits = self.add_weight(
            name="a_logits",
            shape=(),
            initializer=tf.constant_initializer(a_logit),
            trainable=self.a_trainable,
            dtype=self.dtype,
        )
        self.b_logits = self.add_weight(
            name="b_logits",
            shape=(),
            initializer=tf.constant_initializer(b_logit),
            trainable=self.b_trainable,
            dtype=self.dtype,
        )
        self.c_logits = self.add_weight(
            name="c_logits",
            shape=(),
            initializer=tf.constant_initializer(c_logit),
            trainable=self.c_trainable,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        """Compute Big q-Jacobi polynomial basis using three-term recurrence."""
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        
        q = self._get_q(tf.float64)
        a = softplus_lower_bound(tf.cast(self.a_logits, tf.float64), 0.0)
        b = softplus_lower_bound(tf.cast(self.b_logits, tf.float64), 0.0)
        c = -softplus_lower_bound(tf.cast(self.c_logits, tf.float64), 0.0)  # c < 0
        
        ones = tf.ones_like(x, dtype=tf.float64)
        
        # P_0(x) = 1
        P0 = ones
        basis = [P0]
        
        if self.degree >= 1:
            # Three-term recurrence for Big q-Jacobi
            # Coefficients from Koekoek et al.
            
            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for Big q-Jacobi."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)
                ab_q = a * b * q
                
                # A_n (coefficient of P_{n+1})
                numer_A = (1.0 - a * q_n * q) * (1.0 - c * q_n * q) * \
                          (1.0 - ab_q * tf.pow(q, 2.0 * n_f + 1.0))
                denom_A = (1.0 - ab_q * tf.pow(q, 2.0 * n_f + 1.0)) * \
                          (1.0 - ab_q * tf.pow(q, 2.0 * n_f + 2.0))
                A_n = numer_A / (denom_A + 1e-12)
                
                # C_n (coefficient of P_{n-1})
                numer_C = -a * c * q * q_n * (1.0 - q_n) * (1.0 - b * q_n) * \
                          (1.0 - ab_q * q_n / c)
                denom_C = (1.0 - ab_q * tf.pow(q, 2.0 * n_f)) * \
                          (1.0 - ab_q * tf.pow(q, 2.0 * n_f + 1.0))
                C_n = numer_C / (denom_C + 1e-12)
                
                # B_n (coefficient of P_n) - from normalization
                B_n = a * q + c * q - A_n - C_n
                
                return A_n, B_n, C_n
            
            # P_1(x)
            A0, B0, C0 = compute_coeffs(0)
            P1 = (x - B0 * ones) / (A0 + 1e-12)
            basis.append(P1)
            
            # P_n for n >= 2
            P_prev = P0
            P_curr = P1
            
            for n in range(1, self.degree):
                A_n, B_n, C_n = compute_coeffs(n)
                P_next = (x * P_curr - B_n * P_curr - C_n * P_prev) / (A_n + 1e-12)
                basis.append(P_next)
                P_prev = P_curr
                P_curr = P_next
        
        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "a": self.a_init,
            "b": self.b_init,
            "c": self.c_init,
            "a_trainable": self.a_trainable,
            "b_trainable": self.b_trainable,
            "c_trainable": self.c_trainable,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="LittleQJacobi")
class LittleQJacobi(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Little q-Jacobi polynomials.

    The Little q-Jacobi polynomials :math:`p_n(x; a, b; q)` are defined as:

    .. math::

        p_n(x; a, b; q) = {}_2\phi_1\left(\begin{array}{c} 
            q^{-n}, abq^{n+1} \\ aq \end{array}; q, qx\right)

    They are orthogonal with respect to a discrete measure on :math:`\{q^k\}_{k=0}^{\infty}`.

    Three-term recurrence:

    .. math::

        x p_n(x) = A_n p_{n+1}(x) + B_n p_n(x) + C_n p_{n-1}(x)

    Parameters
    ----------
    a : float
        Parameter, 0 < a < q^{-1}.
    b : float
        Parameter, 0 < b < q^{-1}.
    q : float
        Base parameter in (0, 1).

    Notes
    -----
    Little q-Jacobi polynomials with b=0 are called Little q-Laguerre or Wall polynomials.
    In the limit q → 1, these reduce to Jacobi polynomials.

    References
    ----------
    .. [1] NIST DLMF 18.27(iv) - Little q-Jacobi Polynomials
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        a: float = 0.5,
        b: float = 0.5,
        a_trainable: bool = False,
        b_trainable: bool = False,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )
        
        self.a_init = a
        self.b_init = b
        self.a_trainable = a_trainable
        self.b_trainable = b_trainable
        
        self.a_logits = None
        self.b_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        
        # a, b > 0 via softplus
        a_logit = _inverse_softplus_lower_bound(self.a_init, 0.0)
        b_logit = _inverse_softplus_lower_bound(self.b_init, 0.0)
        
        self.a_logits = self.add_weight(
            name="a_logits",
            shape=(),
            initializer=tf.constant_initializer(a_logit),
            trainable=self.a_trainable,
            dtype=self.dtype,
        )
        self.b_logits = self.add_weight(
            name="b_logits",
            shape=(),
            initializer=tf.constant_initializer(b_logit),
            trainable=self.b_trainable,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        """Compute Little q-Jacobi polynomial basis using three-term recurrence."""
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        
        q = self._get_q(tf.float64)
        a = softplus_lower_bound(tf.cast(self.a_logits, tf.float64), 0.0)
        b = softplus_lower_bound(tf.cast(self.b_logits, tf.float64), 0.0)
        
        ones = tf.ones_like(x, dtype=tf.float64)
        
        # p_0(x) = 1
        p0 = ones
        basis = [p0]
        
        if self.degree >= 1:
            # Three-term recurrence for Little q-Jacobi
            # From Koekoek et al., the recurrence is:
            # x p_n = A_n p_{n+1} + B_n p_n + C_n p_{n-1}
            
            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for Little q-Jacobi."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)
                ab_q = a * b * q
                
                # A_n
                numer_A = (1.0 - a * q_n * q) * (1.0 - ab_q * tf.pow(q, 2.0 * n_f + 1.0))
                denom_A = (1.0 - ab_q * tf.pow(q, 2.0 * n_f + 1.0)) * \
                          (1.0 - ab_q * tf.pow(q, 2.0 * n_f + 2.0))
                A_n = numer_A / (denom_A + 1e-12)
                
                # C_n
                numer_C = a * q * q_n * (1.0 - q_n) * (1.0 - b * q_n)
                denom_C = (1.0 - ab_q * tf.pow(q, 2.0 * n_f)) * \
                          (1.0 - ab_q * tf.pow(q, 2.0 * n_f + 1.0))
                C_n = numer_C / (denom_C + 1e-12)
                
                # B_n
                B_n = a * q - A_n - C_n
                
                return A_n, B_n, C_n
            
            # p_1(x)
            A0, B0, C0 = compute_coeffs(0)
            p1 = (x - B0 * ones) / (A0 + 1e-12)
            basis.append(p1)
            
            # p_n for n >= 2
            p_prev = p0
            p_curr = p1
            
            for n in range(1, self.degree):
                A_n, B_n, C_n = compute_coeffs(n)
                p_next = (x * p_curr - B_n * p_curr - C_n * p_prev) / (A_n + 1e-12)
                basis.append(p_next)
                p_prev = p_curr
                p_curr = p_next
        
        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "a": self.a_init,
            "b": self.b_init,
            "a_trainable": self.a_trainable,
            "b_trainable": self.b_trainable,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="QMeixner")
class QMeixner(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using q-Meixner polynomials.

    The q-Meixner polynomials :math:`M_n(x; b, c; q)` are the q-analog of 
    classical Meixner polynomials. They are defined via:

    .. math::

        M_n(q^{-x}; b, c; q) = {}_2\phi_1\left(\begin{array}{c} 
            q^{-n}, q^{-x} \\ bq \end{array}; q, -\frac{q^{n+1}}{c}\right)

    They are orthogonal on :math:`x \in \{0, 1, 2, \ldots\}` with a q-negative
    binomial weight.

    Three-term recurrence:

    .. math::

        x M_n(x) = A_n M_{n+1}(x) + B_n M_n(x) + C_n M_{n-1}(x)

    Parameters
    ----------
    b : float
        Parameter b > 0.
    c : float
        Parameter c > 0.
    q : float
        Base parameter in (0, 1).

    Notes
    -----
    In the limit q → 1, q-Meixner polynomials reduce to classical Meixner polynomials.

    References
    ----------
    .. [1] Koekoek et al. (2010), Chapter 14.11
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        b: float = 1.0,
        c: float = 0.5,
        b_trainable: bool = False,
        c_trainable: bool = False,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )
        
        self.b_init = b
        self.c_init = c
        self.b_trainable = b_trainable
        self.c_trainable = c_trainable
        
        self.b_logits = None
        self.c_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        
        # b, c > 0 via softplus
        b_logit = _inverse_softplus_lower_bound(self.b_init, 0.0)
        c_logit = _inverse_softplus_lower_bound(self.c_init, 0.0)
        
        self.b_logits = self.add_weight(
            name="b_logits",
            shape=(),
            initializer=tf.constant_initializer(b_logit),
            trainable=self.b_trainable,
            dtype=self.dtype,
        )
        self.c_logits = self.add_weight(
            name="c_logits",
            shape=(),
            initializer=tf.constant_initializer(c_logit),
            trainable=self.c_trainable,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        """Compute q-Meixner polynomial basis using three-term recurrence."""
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        
        q = self._get_q(tf.float64)
        b = softplus_lower_bound(tf.cast(self.b_logits, tf.float64), 0.0)
        c = softplus_lower_bound(tf.cast(self.c_logits, tf.float64), 0.0)
        
        ones = tf.ones_like(x, dtype=tf.float64)
        
        # M_0(x) = 1
        M0 = ones
        basis = [M0]
        
        if self.degree >= 1:
            # Three-term recurrence for q-Meixner
            
            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for q-Meixner."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)
                
                # A_n (coefficient of M_{n+1})
                A_n = -c * (1.0 - b * q_n * q) / (1.0 + c)
                
                # C_n (coefficient of M_{n-1})
                C_n = (1.0 - q_n) * (1.0 + c * q_n) / (1.0 + c)
                
                # B_n
                B_n = 1.0 + c * b * q - A_n - C_n
                
                return A_n, B_n, C_n
            
            # M_1(x)
            A0, B0, C0 = compute_coeffs(0)
            M1 = (x - B0 * ones) / (A0 + 1e-12)
            basis.append(M1)
            
            # M_n for n >= 2
            M_prev = M0
            M_curr = M1
            
            for n in range(1, self.degree):
                A_n, B_n, C_n = compute_coeffs(n)
                M_next = (x * M_curr - B_n * M_curr - C_n * M_prev) / (A_n + 1e-12)
                basis.append(M_next)
                M_prev = M_curr
                M_curr = M_next
        
        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "b": self.b_init,
            "c": self.c_init,
            "b_trainable": self.b_trainable,
            "c_trainable": self.c_trainable,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="QKrawtchouk")
class QKrawtchouk(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using q-Krawtchouk polynomials.

    The q-Krawtchouk polynomials :math:`K_n(x; p, N; q)` are the q-analog of
    classical Krawtchouk polynomials. They are defined for :math:`n = 0, 1, \ldots, N`.

    .. math::

        K_n(q^{-x}; p, N; q) = {}_3\phi_2\left(\begin{array}{c}
            q^{-n}, q^{-x}, -pq^n \\ q^{-N}, 0 \end{array}; q, q\right)

    They are orthogonal on :math:`x \in \{0, 1, \ldots, N\}` with a q-binomial weight.

    Three-term recurrence:

    .. math::

        q^{-x} K_n(x) = A_n K_{n+1}(x) + B_n K_n(x) + C_n K_{n-1}(x)

    Parameters
    ----------
    p : float
        Parameter p > 0.
    N : int
        Discrete support size, must be >= degree.
    q : float
        Base parameter in (0, 1).

    Notes
    -----
    In the limit q → 1, q-Krawtchouk polynomials reduce to classical Krawtchouk polynomials.

    References
    ----------
    .. [1] Koekoek et al. (2010), Chapter 14.15
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        p: float = 0.5,
        N: int = 10,
        p_trainable: bool = False,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        if N < degree:
            raise ValueError(f"N must be >= degree, got N={N}, degree={degree}")
        
        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )
        
        self.p_init = p
        self.N = N
        self.p_trainable = p_trainable
        
        self.p_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        
        # p > 0 via softplus
        p_logit = _inverse_softplus_lower_bound(self.p_init, 0.0)
        
        self.p_logits = self.add_weight(
            name="p_logits",
            shape=(),
            initializer=tf.constant_initializer(p_logit),
            trainable=self.p_trainable,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        """Compute q-Krawtchouk polynomial basis using three-term recurrence."""
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        
        q = self._get_q(tf.float64)
        p = softplus_lower_bound(tf.cast(self.p_logits, tf.float64), 0.0)
        N = tf.cast(self.N, tf.float64)
        
        ones = tf.ones_like(x, dtype=tf.float64)
        
        # K_0(x) = 1
        K0 = ones
        basis = [K0]
        
        if self.degree >= 1:
            # Three-term recurrence for q-Krawtchouk
            
            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for q-Krawtchouk."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)
                q_N = tf.pow(q, N)
                
                # A_n (coefficient of K_{n+1})
                A_n = (1.0 - tf.pow(q, n_f - N)) / (1.0 + p * q_n)
                
                # C_n (coefficient of K_{n-1})
                C_n = p * q_n * (1.0 - q_n) / (1.0 + p * tf.pow(q, n_f - 1.0))
                
                # B_n
                B_n = 1.0 - A_n - C_n
                
                return A_n, B_n, C_n
            
            # K_1(x)
            A0, B0, C0 = compute_coeffs(0)
            K1 = (x - B0 * ones) / (A0 + 1e-12)
            basis.append(K1)
            
            # K_n for n >= 2
            K_prev = K0
            K_curr = K1
            
            for n in range(1, self.degree):
                A_n, B_n, C_n = compute_coeffs(n)
                K_next = (x * K_curr - B_n * K_curr - C_n * K_prev) / (A_n + 1e-12)
                basis.append(K_next)
                K_prev = K_curr
                K_curr = K_next
        
        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "p": self.p_init,
            "N": self.N,
            "p_trainable": self.p_trainable,
        })
        return config


# =============================================================================
# Sprint 7F: q-Polynomials Part 2
# =============================================================================


@tfk.utils.register_keras_serializable(package="arnold", name="QCharlier")
class QCharlier(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using q-Charlier polynomials.

    The q-Charlier polynomials :math:`C_n(x; a; q)` are the q-analog of
    classical Charlier polynomials. They are defined as:

    .. math::

        C_n(q^{-x}; a; q) = {}_2\phi_1\left(\begin{array}{c}
            q^{-n}, q^{-x} \\ 0 \end{array}; q, -\frac{q^{n+1}}{a}\right)

    They are orthogonal on :math:`x \in \{0, 1, 2, \ldots\}` with a q-Poisson weight.

    Three-term recurrence:

    .. math::

        q^{-x} C_n(x) = A_n C_{n+1}(x) + B_n C_n(x) + C_n C_{n-1}(x)

    Parameters
    ----------
    a : float
        Parameter a > 0.
    q : float
        Base parameter in (0, 1).

    Notes
    -----
    In the limit q → 1, q-Charlier polynomials reduce to classical Charlier polynomials.
    q-Charlier polynomials are a limit case of q-Meixner with b → 0.

    References
    ----------
    .. [1] Koekoek et al. (2010), Chapter 14.12 - q-Charlier polynomials
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        a: float = 1.0,
        a_trainable: bool = False,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )
        
        self.a_init = a
        self.a_trainable = a_trainable
        self.a_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        
        # a > 0 via softplus
        a_logit = _inverse_softplus_lower_bound(self.a_init, 0.0)
        
        self.a_logits = self.add_weight(
            name="a_logits",
            shape=(),
            initializer=tf.constant_initializer(a_logit),
            trainable=self.a_trainable,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        """Compute q-Charlier polynomial basis using three-term recurrence."""
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        
        q = self._get_q(tf.float64)
        a = softplus_lower_bound(tf.cast(self.a_logits, tf.float64), 0.0)
        
        ones = tf.ones_like(x, dtype=tf.float64)
        
        # C_0(x) = 1
        C0 = ones
        basis = [C0]
        
        if self.degree >= 1:
            # Three-term recurrence for q-Charlier
            # From Koekoek et al., limit of q-Meixner as b → 0
            
            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for q-Charlier."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)
                
                # A_n (coefficient of C_{n+1})
                A_n = -a * q_n * q
                
                # C_n (coefficient of C_{n-1})
                C_n = (1.0 - q_n)
                
                # B_n = 1 + a*q - A_n - C_n
                B_n = 1.0 + a * q - A_n - C_n
                
                return A_n, B_n, C_n
            
            # C_1(x)
            A0, B0, _ = compute_coeffs(0)
            C1 = (x - B0 * ones) / (A0 + 1e-12)
            basis.append(C1)
            
            # C_n for n >= 2
            C_prev = C0
            C_curr = C1
            
            for n in range(1, self.degree):
                A_n, B_n, C_coeff = compute_coeffs(n)
                C_next = (x * C_curr - B_n * C_curr - C_coeff * C_prev) / (A_n + 1e-12)
                basis.append(C_next)
                C_prev = C_curr
                C_curr = C_next
        
        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "a": self.a_init,
            "a_trainable": self.a_trainable,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="QRacah")
class QRacah(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using q-Racah polynomials.

    The q-Racah polynomials :math:`R_n(x; \alpha, \beta, \gamma, \delta; q)` are
    the top of the Askey-Wilson scheme for discrete q-orthogonal polynomials.
    They are defined for :math:`n = 0, 1, \ldots, N` where one of 
    :math:`\alpha q, \beta\delta q, \gamma q = q^{-N}`.

    .. math::

        R_n(\mu(y); \alpha, \beta, \gamma, \delta | q) = 
            {}_4\phi_3\left(\begin{array}{c}
            q^{-n}, \alpha\beta q^{n+1}, q^{-y}, \gamma\delta q^{y+1} \\
            \alpha q, \beta\delta q, \gamma q \end{array}; q, q\right)

    where :math:`\mu(y) = q^{-y} + \gamma\delta q^{y+1}`.

    Three-term recurrence:

    .. math::

        \mu(y) R_n(y) = A_n R_{n+1}(y) + B_n R_n(y) + C_n R_{n-1}(y)

    Parameters
    ----------
    alpha : float
        Parameter α > 0.
    beta : float
        Parameter β > 0.
    gamma : float
        Parameter γ > 0.
    delta : float
        Parameter δ > 0.
    N : int
        Discrete support size, must be >= degree.
    q : float
        Base parameter in (0, 1).

    Notes
    -----
    q-Racah polynomials satisfy a duality relation: swapping (α, β) ↔ (γ, δ) 
    and n ↔ y gives the same polynomial value.
    In the limit q → 1, q-Racah polynomials reduce to classical Racah polynomials.

    References
    ----------
    .. [1] NIST DLMF 18.28(viii) - q-Racah Polynomials
    .. [2] Koekoek et al. (2010), Chapter 14.2
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        delta: float = 1.0,
        N: int = 10,
        alpha_trainable: bool = False,
        beta_trainable: bool = False,
        gamma_trainable: bool = False,
        delta_trainable: bool = False,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        if N < degree:
            raise ValueError(f"N must be >= degree, got N={N}, degree={degree}")
        
        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )
        
        self.alpha_init = alpha
        self.beta_init = beta
        self.gamma_init = gamma
        self.delta_init = delta
        self.N = N
        self.alpha_trainable = alpha_trainable
        self.beta_trainable = beta_trainable
        self.gamma_trainable = gamma_trainable
        self.delta_trainable = delta_trainable
        
        self.alpha_logits = None
        self.beta_logits = None
        self.gamma_logits = None
        self.delta_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        
        # All params > 0 via softplus
        alpha_logit = _inverse_softplus_lower_bound(self.alpha_init, 0.0)
        beta_logit = _inverse_softplus_lower_bound(self.beta_init, 0.0)
        gamma_logit = _inverse_softplus_lower_bound(self.gamma_init, 0.0)
        delta_logit = _inverse_softplus_lower_bound(self.delta_init, 0.0)
        
        self.alpha_logits = self.add_weight(
            name="alpha_logits",
            shape=(),
            initializer=tf.constant_initializer(alpha_logit),
            trainable=self.alpha_trainable,
            dtype=self.dtype,
        )
        self.beta_logits = self.add_weight(
            name="beta_logits",
            shape=(),
            initializer=tf.constant_initializer(beta_logit),
            trainable=self.beta_trainable,
            dtype=self.dtype,
        )
        self.gamma_logits = self.add_weight(
            name="gamma_logits",
            shape=(),
            initializer=tf.constant_initializer(gamma_logit),
            trainable=self.gamma_trainable,
            dtype=self.dtype,
        )
        self.delta_logits = self.add_weight(
            name="delta_logits",
            shape=(),
            initializer=tf.constant_initializer(delta_logit),
            trainable=self.delta_trainable,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        """Compute q-Racah polynomial basis using three-term recurrence."""
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        
        q = self._get_q(tf.float64)
        alpha = softplus_lower_bound(tf.cast(self.alpha_logits, tf.float64), 0.0)
        beta = softplus_lower_bound(tf.cast(self.beta_logits, tf.float64), 0.0)
        gamma = softplus_lower_bound(tf.cast(self.gamma_logits, tf.float64), 0.0)
        delta = softplus_lower_bound(tf.cast(self.delta_logits, tf.float64), 0.0)
        N = tf.cast(self.N, tf.float64)
        
        ones = tf.ones_like(x, dtype=tf.float64)
        
        # R_0(x) = 1
        R0 = ones
        basis = [R0]
        
        if self.degree >= 1:
            # Three-term recurrence for q-Racah
            # Coefficients from Koekoek et al. (2010)
            
            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for q-Racah."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)
                ab_q = alpha * beta * q
                gd_q = gamma * delta * q
                
                # Denominator terms
                denom1 = 1.0 - ab_q * tf.pow(q, 2.0 * n_f + 1.0)
                denom2 = 1.0 - ab_q * tf.pow(q, 2.0 * n_f + 2.0)
                denom3 = 1.0 - ab_q * tf.pow(q, 2.0 * n_f)
                
                # A_n (coefficient of R_{n+1})
                numer_A = (1.0 - alpha * q_n * q) * (1.0 - beta * delta * q_n * q) * \
                          (1.0 - gamma * q_n * q) * (1.0 - ab_q * tf.pow(q, n_f + N + 1.0))
                A_n = numer_A / (denom1 * denom2 + 1e-12)
                
                # C_n (coefficient of R_{n-1})
                numer_C = (1.0 - q_n) * (alpha - gamma * q_n) * \
                          (beta * delta - gamma * q_n) * \
                          (1.0 - gd_q * tf.pow(q, n_f + N))
                C_n = -alpha * gamma * q * numer_C / ((denom1 * denom3 + 1e-12) * ab_q)
                
                # B_n = (1 + gd*q) - A_n - C_n
                B_n = 1.0 + gd_q - A_n - C_n
                
                return A_n, B_n, C_n
            
            # R_1(x)
            A0, B0, _ = compute_coeffs(0)
            R1 = (x - B0 * ones) / (A0 + 1e-12)
            basis.append(R1)
            
            # R_n for n >= 2
            R_prev = R0
            R_curr = R1
            
            for n in range(1, self.degree):
                A_n, B_n, C_n = compute_coeffs(n)
                R_next = (x * R_curr - B_n * R_curr - C_n * R_prev) / (A_n + 1e-12)
                basis.append(R_next)
                R_prev = R_curr
                R_curr = R_next
        
        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha_init,
            "beta": self.beta_init,
            "gamma": self.gamma_init,
            "delta": self.delta_init,
            "N": self.N,
            "alpha_trainable": self.alpha_trainable,
            "beta_trainable": self.beta_trainable,
            "gamma_trainable": self.gamma_trainable,
            "delta_trainable": self.delta_trainable,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="DualQHahn")
class DualQHahn(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Dual q-Hahn polynomials.

    The Dual q-Hahn polynomials :math:`R_n(\mu(x); \gamma, \delta, N; q)` are
    dual to the q-Hahn polynomials. They are defined for :math:`n = 0, 1, \ldots, N`.

    .. math::

        R_n(\mu(x); \gamma, \delta, N | q) = 
            {}_3\phi_2\left(\begin{array}{c}
            q^{-n}, q^{-x}, \gamma\delta q^{x+1} \\
            \gamma q, q^{-N} \end{array}; q, q\right)

    where :math:`\mu(x) = q^{-x} + \gamma\delta q^{x+1}`.

    Three-term recurrence:

    .. math::

        \mu(x) R_n(x) = A_n R_{n+1}(x) + B_n R_n(x) + C_n R_{n-1}(x)

    Parameters
    ----------
    gamma : float
        Parameter γ > 0.
    delta : float
        Parameter δ > 0.
    N : int
        Discrete support size, must be >= degree.
    q : float
        Base parameter in (0, 1).

    Notes
    -----
    Dual q-Hahn polynomials are limit cases of q-Racah with α = 0.
    In the limit q → 1, they reduce to dual Hahn polynomials.

    References
    ----------
    .. [1] Koekoek et al. (2010), Chapter 14.7 - Dual q-Hahn polynomials
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        gamma: float = 1.0,
        delta: float = 1.0,
        N: int = 10,
        gamma_trainable: bool = False,
        delta_trainable: bool = False,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        if N < degree:
            raise ValueError(f"N must be >= degree, got N={N}, degree={degree}")
        
        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )
        
        self.gamma_init = gamma
        self.delta_init = delta
        self.N = N
        self.gamma_trainable = gamma_trainable
        self.delta_trainable = delta_trainable
        
        self.gamma_logits = None
        self.delta_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        
        gamma_logit = _inverse_softplus_lower_bound(self.gamma_init, 0.0)
        delta_logit = _inverse_softplus_lower_bound(self.delta_init, 0.0)
        
        self.gamma_logits = self.add_weight(
            name="gamma_logits",
            shape=(),
            initializer=tf.constant_initializer(gamma_logit),
            trainable=self.gamma_trainable,
            dtype=self.dtype,
        )
        self.delta_logits = self.add_weight(
            name="delta_logits",
            shape=(),
            initializer=tf.constant_initializer(delta_logit),
            trainable=self.delta_trainable,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        """Compute Dual q-Hahn polynomial basis using three-term recurrence."""
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        
        q = self._get_q(tf.float64)
        gamma = softplus_lower_bound(tf.cast(self.gamma_logits, tf.float64), 0.0)
        delta = softplus_lower_bound(tf.cast(self.delta_logits, tf.float64), 0.0)
        N = tf.cast(self.N, tf.float64)
        
        ones = tf.ones_like(x, dtype=tf.float64)
        
        # R_0(x) = 1
        R0 = ones
        basis = [R0]
        
        if self.degree >= 1:
            # Three-term recurrence for Dual q-Hahn
            gd_q = gamma * delta * q
            
            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for Dual q-Hahn."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)
                
                # A_n (coefficient of R_{n+1})
                A_n = (1.0 - gamma * q_n * q) * (1.0 - tf.pow(q, n_f - N))
                
                # C_n (coefficient of R_{n-1})  
                C_n = gamma * q_n * (1.0 - q_n) * (delta - tf.pow(q, n_f - N - 1.0))
                
                # B_n = 1 + gd*q - A_n - C_n
                B_n = 1.0 + gd_q - A_n - C_n
                
                return A_n, B_n, C_n
            
            # R_1(x)
            A0, B0, _ = compute_coeffs(0)
            R1 = (x - B0 * ones) / (A0 + 1e-12)
            basis.append(R1)
            
            # R_n for n >= 2
            R_prev = R0
            R_curr = R1
            
            for n in range(1, self.degree):
                A_n, B_n, C_n = compute_coeffs(n)
                R_next = (x * R_curr - B_n * R_curr - C_n * R_prev) / (A_n + 1e-12)
                basis.append(R_next)
                R_prev = R_curr
                R_curr = R_next
        
        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma_init,
            "delta": self.delta_init,
            "N": self.N,
            "gamma_trainable": self.gamma_trainable,
            "delta_trainable": self.delta_trainable,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="DualQKrawtchouk")
class DualQKrawtchouk(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Dual q-Krawtchouk polynomials.

    The Dual q-Krawtchouk polynomials :math:`K_n(\lambda(x); c, N; q)` are
    dual to the q-Krawtchouk polynomials. They are defined for :math:`n = 0, 1, \ldots, N`.

    .. math::

        K_n(\lambda(x); c, N | q) = 
            {}_3\phi_2\left(\begin{array}{c}
            q^{-n}, q^{-x}, cq^{x-N} \\
            q^{-N}, 0 \end{array}; q, q\right)

    where :math:`\lambda(x) = q^{-x} + c q^{x-N}`.

    Three-term recurrence:

    .. math::

        \lambda(x) K_n(x) = A_n K_{n+1}(x) + B_n K_n(x) + C_n K_{n-1}(x)

    Parameters
    ----------
    c : float
        Parameter c > 0.
    N : int
        Discrete support size, must be >= degree.
    q : float
        Base parameter in (0, 1).

    Notes
    -----
    Dual q-Krawtchouk polynomials are limit cases of Dual q-Hahn with δ → 0.
    In the limit q → 1, they reduce to dual Krawtchouk polynomials.

    References
    ----------
    .. [1] Koekoek et al. (2010), Chapter 14.17 - Dual q-Krawtchouk polynomials
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        c: float = 1.0,
        N: int = 10,
        c_trainable: bool = False,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        if N < degree:
            raise ValueError(f"N must be >= degree, got N={N}, degree={degree}")
        
        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )
        
        self.c_init = c
        self.N = N
        self.c_trainable = c_trainable
        self.c_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        
        c_logit = _inverse_softplus_lower_bound(self.c_init, 0.0)
        
        self.c_logits = self.add_weight(
            name="c_logits",
            shape=(),
            initializer=tf.constant_initializer(c_logit),
            trainable=self.c_trainable,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        """Compute Dual q-Krawtchouk polynomial basis using three-term recurrence."""
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        
        q = self._get_q(tf.float64)
        c = softplus_lower_bound(tf.cast(self.c_logits, tf.float64), 0.0)
        N = tf.cast(self.N, tf.float64)
        
        ones = tf.ones_like(x, dtype=tf.float64)
        
        # K_0(x) = 1
        K0 = ones
        basis = [K0]
        
        if self.degree >= 1:
            # Three-term recurrence for Dual q-Krawtchouk
            
            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for Dual q-Krawtchouk."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)
                
                # A_n (coefficient of K_{n+1})
                A_n = 1.0 - tf.pow(q, n_f - N)
                
                # C_n (coefficient of K_{n-1})
                C_n = c * tf.pow(q, n_f - N) * (1.0 - q_n)
                
                # B_n = 1 + c*q^{-N} - A_n - C_n
                B_n = 1.0 + c * tf.pow(q, -N) - A_n - C_n
                
                return A_n, B_n, C_n
            
            # K_1(x)
            A0, B0, _ = compute_coeffs(0)
            K1 = (x - B0 * ones) / (A0 + 1e-12)
            basis.append(K1)
            
            # K_n for n >= 2
            K_prev = K0
            K_curr = K1
            
            for n in range(1, self.degree):
                A_n, B_n, C_n = compute_coeffs(n)
                K_next = (x * K_curr - B_n * K_curr - C_n * K_prev) / (A_n + 1e-12)
                basis.append(K_next)
                K_prev = K_curr
                K_curr = K_next
        
        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "c": self.c_init,
            "N": self.N,
            "c_trainable": self.c_trainable,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="AffineQKrawtchouk")
class AffineQKrawtchouk(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Affine q-Krawtchouk polynomials.

    The Affine q-Krawtchouk polynomials :math:`K_n^{\text{Aff}}(x; p, N; q)` are
    an affine variant of q-Krawtchouk polynomials. They are defined for 
    :math:`n = 0, 1, \ldots, N`.

    .. math::

        K_n^{\text{Aff}}(q^{-x}; p, N | q) = 
            {}_3\phi_2\left(\begin{array}{c}
            q^{-n}, 0, q^{-x} \\
            pq, q^{-N} \end{array}; q, q\right)

    They are orthogonal on :math:`x \in \{0, 1, \ldots, N\}` with an affine
    q-binomial weight.

    Three-term recurrence:

    .. math::

        q^{-x} K_n(x) = A_n K_{n+1}(x) + B_n K_n(x) + C_n K_{n-1}(x)

    Parameters
    ----------
    p : float
        Parameter p > 0.
    N : int
        Discrete support size, must be >= degree.
    q : float
        Base parameter in (0, 1).

    Notes
    -----
    Affine q-Krawtchouk polynomials are a special case of the affine q-analog
    of the classical Krawtchouk polynomials, obtained in the limit β → 0 of
    q-Hahn polynomials.

    References
    ----------
    .. [1] Koekoek et al. (2010), Chapter 14.16 - Affine q-Krawtchouk polynomials
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        p: float = 0.5,
        N: int = 10,
        p_trainable: bool = False,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        if N < degree:
            raise ValueError(f"N must be >= degree, got N={N}, degree={degree}")
        
        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )
        
        self.p_init = p
        self.N = N
        self.p_trainable = p_trainable
        self.p_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        
        p_logit = _inverse_softplus_lower_bound(self.p_init, 0.0)
        
        self.p_logits = self.add_weight(
            name="p_logits",
            shape=(),
            initializer=tf.constant_initializer(p_logit),
            trainable=self.p_trainable,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        """Compute Affine q-Krawtchouk polynomial basis using three-term recurrence."""
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        
        q = self._get_q(tf.float64)
        p = softplus_lower_bound(tf.cast(self.p_logits, tf.float64), 0.0)
        N = tf.cast(self.N, tf.float64)
        
        ones = tf.ones_like(x, dtype=tf.float64)
        
        # K_0(x) = 1
        K0 = ones
        basis = [K0]
        
        if self.degree >= 1:
            # Three-term recurrence for Affine q-Krawtchouk
            
            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for Affine q-Krawtchouk."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)
                
                # A_n (coefficient of K_{n+1})
                A_n = (1.0 - p * q_n * q) * (1.0 - tf.pow(q, n_f - N))
                
                # C_n (coefficient of K_{n-1})
                C_n = p * q * (1.0 - q_n) * tf.pow(q, n_f - N - 1.0)
                
                # B_n = 1 + p*q - A_n - C_n
                B_n = 1.0 + p * q - A_n - C_n
                
                return A_n, B_n, C_n
            
            # K_1(x)
            A0, B0, _ = compute_coeffs(0)
            K1 = (x - B0 * ones) / (A0 + 1e-12)
            basis.append(K1)
            
            # K_n for n >= 2
            K_prev = K0
            K_curr = K1
            
            for n in range(1, self.degree):
                A_n, B_n, C_n = compute_coeffs(n)
                K_next = (x * K_curr - B_n * K_curr - C_n * K_prev) / (A_n + 1e-12)
                basis.append(K_next)
                K_prev = K_curr
                K_curr = K_next
        
        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "p": self.p_init,
            "N": self.N,
            "p_trainable": self.p_trainable,
        })
        return config


# =============================================================================
# Sprint 7G: q-Polynomials Part 3 (Askey-Wilson Class)
# =============================================================================


@tfk.utils.register_keras_serializable(package="arnold", name="DiscreteQHermite1")
class DiscreteQHermite1(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Discrete q-Hermite I polynomials.

    The Discrete q-Hermite I polynomials :math:`h_n(x; q)` are defined as:

    .. math::

        h_n(x; q) = x^n \cdot {}_2\phi_0\left(\begin{array}{c}
            q^{-n}, q^{-n+1} \\ - \end{array}; q^2, \frac{q^{2n-1}}{x^2}\right)

    They satisfy the remarkably simple three-term recurrence:

    .. math::

        x \, h_n(x; q) = h_{n+1}(x; q) + (1 - q^n) \, h_{n-1}(x; q)

    with initial conditions :math:`h_0(x; q) = 1` and :math:`h_1(x; q) = x`.

    Orthogonality relation on :math:`\{q^k : k \in \mathbb{Z}\}`:

    .. math::

        \sum_{k=-\infty}^{\infty} h_m(q^k; q) \, h_n(q^k; q) \, q^k = 
            \frac{(q; q)_n \, (q; q)_\infty}{(-q; q)_\infty} \, \delta_{mn}

    Parameters
    ----------
    q : float
        Base parameter in (0, 1). Default is 0.5.
    q_trainable : bool
        Whether q is trainable. Default is True.

    Notes
    -----
    - Discrete q-Hermite I are symmetric: :math:`h_n(-x; q) = (-1)^n h_n(x; q)`
    - The recurrence has no x-dependent coefficients (only n-dependent)
    - In the limit q → 1: :math:`\lim_{q \to 1} h_n(x(1-q^2)^{1/2}; q) / (1-q^2)^{n/2} = 2^{-n} H_n(x)`

    References
    ----------
    .. [1] NIST DLMF 18.27(vii) - Discrete q-Hermite I and II Polynomials
    .. [2] Koekoek et al. (2010), Chapter 14.28
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        **kwargs,
    ):
        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Compute Discrete q-Hermite I polynomial basis.

        Uses the simple three-term recurrence:

        .. math::

            h_{n+1}(x) = x \cdot h_n(x) - (1 - q^n) \cdot h_{n-1}(x)

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, degree+1).
        """
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        q = self._get_q(tf.float64)

        # h_0(x) = 1
        h0 = tf.ones_like(x, dtype=tf.float64)
        basis = [h0]

        if self.degree >= 1:
            # h_1(x) = x
            h1 = x
            basis.append(h1)

            # Recurrence: h_{n+1}(x) = x * h_n(x) - (1 - q^n) * h_{n-1}(x)
            h_prev = h0
            h_curr = h1

            for n in range(1, self.degree):
                q_n = tf.pow(q, tf.cast(n, tf.float64))
                C_n = 1.0 - q_n  # Coefficient of h_{n-1}

                h_next = x * h_curr - C_n * h_prev
                basis.append(h_next)
                h_prev = h_curr
                h_curr = h_next

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)


@tfk.utils.register_keras_serializable(package="arnold", name="DiscreteQHermite2")
class DiscreteQHermite2(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Discrete q-Hermite II polynomials.

    The Discrete q-Hermite II polynomials :math:`\tilde{h}_n(x; q)` are defined as:

    .. math::

        \tilde{h}_n(x; q) = x^n \cdot {}_2\phi_1\left(\begin{array}{c}
            q^{-n}, q^{-n+1} \\ 0 \end{array}; q^2, -\frac{q^2}{x^2}\right)

    They satisfy the three-term recurrence:

    .. math::

        x \, \tilde{h}_n(x; q) = \tilde{h}_{n+1}(x; q) + q^{n-1}(1 - q^n) \, \tilde{h}_{n-1}(x; q)

    with initial conditions :math:`\tilde{h}_0(x; q) = 1` and :math:`\tilde{h}_1(x; q) = x`.

    Orthogonality relation (non-unique measure):

    .. math::

        \sum_{k=-\infty}^{\infty} \tilde{h}_m(cq^k; q) \, \tilde{h}_n(cq^k; q) \, 
            \frac{q^{k(k-1)}}{(-c^2 q^{2k-1}; q^2)_\infty} = h_n \, \delta_{mn}

    Parameters
    ----------
    q : float
        Base parameter in (0, 1). Default is 0.5.
    q_trainable : bool
        Whether q is trainable. Default is True.

    Notes
    -----
    - Discrete q-Hermite II are also symmetric: :math:`\tilde{h}_n(-x; q) = (-1)^n \tilde{h}_n(x; q)`
    - The measure is not uniquely determined (indeterminate moment problem)
    - Same q → 1 limit as Discrete q-Hermite I

    References
    ----------
    .. [1] NIST DLMF 18.27(vii) - Discrete q-Hermite I and II Polynomials
    .. [2] Koekoek et al. (2010), Chapter 14.29
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        **kwargs,
    ):
        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Compute Discrete q-Hermite II polynomial basis.

        Uses the three-term recurrence:

        .. math::

            \tilde{h}_{n+1}(x) = x \cdot \tilde{h}_n(x) - q^{n-1}(1 - q^n) \cdot \tilde{h}_{n-1}(x)

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, degree+1).
        """
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        q = self._get_q(tf.float64)

        # h̃_0(x) = 1
        h0 = tf.ones_like(x, dtype=tf.float64)
        basis = [h0]

        if self.degree >= 1:
            # h̃_1(x) = x
            h1 = x
            basis.append(h1)

            # Recurrence: h̃_{n+1}(x) = x * h̃_n(x) - q^{n-1}(1 - q^n) * h̃_{n-1}(x)
            h_prev = h0
            h_curr = h1

            for n in range(1, self.degree):
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)
                q_nm1 = tf.pow(q, n_f - 1.0)
                C_n = q_nm1 * (1.0 - q_n)  # Coefficient of h̃_{n-1}

                h_next = x * h_curr - C_n * h_prev
                basis.append(h_next)
                h_prev = h_curr
                h_curr = h_next

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)


@tfk.utils.register_keras_serializable(package="arnold", name="ContinuousQHermite")
class ContinuousQHermite(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Continuous q-Hermite polynomials.

    The Continuous q-Hermite polynomials :math:`H_n(x | q)` are defined as:

    .. math::

        H_n(\cos\theta | q) = \sum_{\ell=0}^{n} \frac{(q; q)_n}{(q; q)_\ell (q; q)_{n-\ell}} 
            e^{i(n-2\ell)\theta}

    They satisfy the three-term recurrence:

    .. math::

        2x \, H_n(x | q) = H_{n+1}(x | q) + (1 - q^n) \, H_{n-1}(x | q)

    with :math:`H_0(x | q) = 1` and :math:`H_1(x | q) = 2x`.

    Orthogonality on :math:`[-1, 1]`:

    .. math::

        \frac{1}{2\pi} \int_0^\pi H_m(\cos\theta | q) \, H_n(\cos\theta | q) \,
            (e^{2i\theta}; q)_\infty (e^{-2i\theta}; q)_\infty \, d\theta = 
            \frac{(q; q)_n}{(q; q)_\infty} \, \delta_{mn}

    Parameters
    ----------
    q : float
        Base parameter in (0, 1). Default is 0.5.
    q_trainable : bool
        Whether q is trainable. Default is True.

    Notes
    -----
    - Also known as Rogers-Szegő polynomials
    - Limit to Hermite: :math:`\lim_{q \to 1} H_n(x\sqrt{(1-q)/2} | q) / ((1-q)/2)^{n/2} = H_n(x)`
    - The weight function is real and positive on [-1, 1]

    References
    ----------
    .. [1] NIST DLMF 18.28(vi) - Continuous q-Hermite Polynomials
    .. [2] Koekoek et al. (2010), Chapter 14.26
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        **kwargs,
    ):
        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Compute Continuous q-Hermite polynomial basis.

        Uses the three-term recurrence:

        .. math::

            H_{n+1}(x) = 2x \cdot H_n(x) - (1 - q^n) \cdot H_{n-1}(x)

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, degree+1).
        """
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        q = self._get_q(tf.float64)

        # H_0(x) = 1
        H0 = tf.ones_like(x, dtype=tf.float64)
        basis = [H0]

        if self.degree >= 1:
            # H_1(x) = 2x
            H1 = 2.0 * x
            basis.append(H1)

            # Recurrence: H_{n+1}(x) = 2x * H_n(x) - (1 - q^n) * H_{n-1}(x)
            H_prev = H0
            H_curr = H1

            for n in range(1, self.degree):
                q_n = tf.pow(q, tf.cast(n, tf.float64))
                C_n = 1.0 - q_n  # Coefficient of H_{n-1}

                H_next = 2.0 * x * H_curr - C_n * H_prev
                basis.append(H_next)
                H_prev = H_curr
                H_curr = H_next

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)


@tfk.utils.register_keras_serializable(package="arnold", name="ContinuousQJacobi")
class ContinuousQJacobi(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Continuous q-Jacobi polynomials.

    The Continuous q-Jacobi polynomials :math:`P_n^{(\alpha,\beta)}(x | q)` are defined as:

    .. math::

        P_n^{(\alpha,\beta)}(x | q) = \frac{(q^{\alpha+1}; q)_n}{(q; q)_n} \,
            {}_4\phi_3\left(\begin{array}{c}
            q^{-n}, q^{n+\alpha+\beta+1}, q^{\alpha/2+1/4} e^{i\theta}, q^{\alpha/2+1/4} e^{-i\theta} \\
            q^{\alpha+1}, -q^{(\alpha+\beta+1)/2}, -q^{(\alpha+\beta+2)/2}
            \end{array}; q, q\right)

    where :math:`x = \cos\theta`.

    They satisfy a three-term recurrence of the form:

    .. math::

        x \, P_n(x) = A_n P_{n+1}(x) + B_n P_n(x) + C_n P_{n-1}(x)

    Parameters
    ----------
    alpha : float
        Parameter α > -1. Default is 0.0.
    beta : float
        Parameter β > -1. Default is 0.0.
    alpha_trainable : bool
        Whether α is trainable. Default is False.
    beta_trainable : bool
        Whether β is trainable. Default is False.
    q : float
        Base parameter in (0, 1). Default is 0.5.

    Notes
    -----
    - Limit to Jacobi: :math:`\lim_{q \to 1} P_n^{(\alpha,\beta)}(x | q) = P_n^{(\alpha,\beta)}(x)`
    - Special case: :math:`P_n^{(\lambda-1/2, \lambda-1/2)}(x | q) \propto C_n(x; q^\lambda | q)` (q-ultraspherical)
    - Orthogonal on [-1, 1] with a q-deformed beta weight

    References
    ----------
    .. [1] NIST DLMF 18.28(ix) - Continuous q-Jacobi Polynomials
    .. [2] Koekoek et al. (2010), Chapter 14.10
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        alpha: float = 0.0,
        beta: float = 0.0,
        alpha_trainable: bool = False,
        beta_trainable: bool = False,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        **kwargs,
    ):
        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )

        self.alpha_init = alpha
        self.beta_init = beta
        self.alpha_trainable = alpha_trainable
        self.beta_trainable = beta_trainable

        self.alpha_logits = None
        self.beta_logits = None

    def build(self, input_shape):
        super().build(input_shape)

        # α, β > -1 via softplus with lower bound -1
        alpha_logit = _inverse_softplus_lower_bound(self.alpha_init + 1.0, 0.0)
        beta_logit = _inverse_softplus_lower_bound(self.beta_init + 1.0, 0.0)

        self.alpha_logits = self.add_weight(
            name="alpha_logits",
            shape=(),
            initializer=tf.constant_initializer(alpha_logit),
            trainable=self.alpha_trainable,
            dtype=self.dtype,
        )
        self.beta_logits = self.add_weight(
            name="beta_logits",
            shape=(),
            initializer=tf.constant_initializer(beta_logit),
            trainable=self.beta_trainable,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Compute Continuous q-Jacobi polynomial basis using three-term recurrence.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, degree+1).
        """
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)

        q = self._get_q(tf.float64)
        # α, β > -1: we store α+1, β+1 > 0
        alpha = softplus_lower_bound(tf.cast(self.alpha_logits, tf.float64), 0.0) - 1.0
        beta = softplus_lower_bound(tf.cast(self.beta_logits, tf.float64), 0.0) - 1.0

        # P_0(x) = 1
        P0 = tf.ones_like(x, dtype=tf.float64)
        basis = [P0]

        if self.degree >= 1:
            # Three-term recurrence coefficients for continuous q-Jacobi
            # Simplified form based on Koekoek et al.

            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for continuous q-Jacobi."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)

                ab = alpha + beta
                q_ab = tf.pow(q, ab)
                q_a = tf.pow(q, alpha)
                q_b = tf.pow(q, beta)

                # Denominator terms
                denom1 = 1.0 - q_ab * tf.pow(q, 2.0 * n_f + 1.0)
                denom2 = 1.0 - q_ab * tf.pow(q, 2.0 * n_f + 2.0)
                denom3 = 1.0 - q_ab * tf.pow(q, 2.0 * n_f)

                # A_n (coefficient of P_{n+1})
                numer_A = (1.0 - q_a * q_n * q) * (1.0 - q_ab * q_n * q)
                A_n = 0.5 * numer_A / (denom1 * denom2 + 1e-12)

                # C_n (coefficient of P_{n-1})
                numer_C = (1.0 - q_n) * (1.0 - q_b * q_n)
                C_n = 0.5 * q_a * q * numer_C / (denom1 * denom3 + 1e-12)

                # B_n
                B_n = 0.5 * (1.0 + q_a * q_b * q) - A_n - C_n

                return A_n, B_n, C_n

            # P_1(x)
            A0, B0, _ = compute_coeffs(0)
            P1 = (x - B0) / (A0 + 1e-12)
            basis.append(P1)

            # P_n for n >= 2
            P_prev = P0
            P_curr = P1

            for n in range(1, self.degree):
                A_n, B_n, C_n = compute_coeffs(n)
                P_next = (x * P_curr - B_n * P_curr - C_n * P_prev) / (A_n + 1e-12)
                basis.append(P_next)
                P_prev = P_curr
                P_curr = P_next

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha_init,
            "beta": self.beta_init,
            "alpha_trainable": self.alpha_trainable,
            "beta_trainable": self.beta_trainable,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="ContinuousQUltraspherical")
class ContinuousQUltraspherical(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Continuous q-Ultraspherical (Rogers) polynomials.

    The Continuous q-Ultraspherical polynomials :math:`C_n(x; \beta | q)` are defined as:

    .. math::

        C_n(\cos\theta; \beta | q) = \sum_{\ell=0}^{n} 
            \frac{(\beta; q)_\ell (\beta; q)_{n-\ell}}{(q; q)_\ell (q; q)_{n-\ell}} 
            e^{i(n-2\ell)\theta}

    They satisfy the three-term recurrence:

    .. math::

        2x(1 - \beta q^n) C_n(x) = (1 - q^{n+1}) C_{n+1}(x) + (1 - \beta^2 q^{n-1}) C_{n-1}(x)

    with :math:`C_0(x; \beta | q) = 1` and :math:`C_1(x; \beta | q) = 2x(1-\beta)/(1-q)`.

    Orthogonality on :math:`[-1, 1]`:

    .. math::

        \frac{1}{2\pi} \int_0^\pi C_m(\cos\theta; \beta | q) \, C_n(\cos\theta; \beta | q) \,
            w(\cos\theta) \, d\theta = h_n \, \delta_{mn}

    Parameters
    ----------
    beta : float
        Parameter β with |β| < 1. Default is 0.5.
    beta_trainable : bool
        Whether β is trainable. Default is False.
    q : float
        Base parameter in (0, 1). Default is 0.5.

    Notes
    -----
    - Also known as Rogers polynomials
    - Limit: :math:`\lim_{q \to 1} C_n(x; q^\lambda | q) = C_n^{(\lambda)}(x)` (Gegenbauer)
    - Special case β = 0: :math:`C_n(x; 0 | q) = H_n(x | q) / (q; q)_n` (q-Hermite)
    - Symmetric: :math:`C_n(-x; \beta | q) = (-1)^n C_n(x; \beta | q)`

    References
    ----------
    .. [1] NIST DLMF 18.28(v) - Continuous q-Ultraspherical Polynomials
    .. [2] Koekoek et al. (2010), Chapter 14.10.1
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        beta: float = 0.5,
        beta_trainable: bool = False,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        **kwargs,
    ):
        if not -1 < beta < 1:
            raise ValueError(f"beta must be in (-1, 1), got {beta}")

        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )

        self.beta_init = beta
        self.beta_trainable = beta_trainable
        self.beta_logits = None

    def build(self, input_shape):
        super().build(input_shape)

        # β ∈ (-1, 1) via scaled tanh: β = tanh(logit)
        # Inverse: logit = arctanh(β)
        beta_logit = 0.5 * math.log((1 + self.beta_init) / (1 - self.beta_init + 1e-12))

        self.beta_logits = self.add_weight(
            name="beta_logits",
            shape=(),
            initializer=tf.constant_initializer(beta_logit),
            trainable=self.beta_trainable,
            dtype=self.dtype,
        )

    def _get_beta(self, dtype=tf.float64):
        """Get β parameter constrained to (-1, 1) via tanh."""
        return tf.tanh(tf.cast(self.beta_logits, dtype))

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Compute Continuous q-Ultraspherical polynomial basis.

        Uses the three-term recurrence:

        .. math::

            2x(1 - \beta q^n) C_n = (1 - q^{n+1}) C_{n+1} + (1 - \beta^2 q^{n-1}) C_{n-1}

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, degree+1).
        """
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)

        q = self._get_q(tf.float64)
        beta = self._get_beta(tf.float64)

        # C_0(x) = 1
        C0 = tf.ones_like(x, dtype=tf.float64)
        basis = [C0]

        if self.degree >= 1:
            # C_1(x) = 2x(1 - β) / (1 - q)
            # For numerical stability, use direct formula
            C1 = 2.0 * x * (1.0 - beta) / (1.0 - q + 1e-12)
            basis.append(C1)

            # Recurrence: 2x(1 - β*q^n) C_n = (1 - q^{n+1}) C_{n+1} + (1 - β²*q^{n-1}) C_{n-1}
            # Rearranged: C_{n+1} = [2x(1 - β*q^n) C_n - (1 - β²*q^{n-1}) C_{n-1}] / (1 - q^{n+1})
            C_prev = C0
            C_curr = C1

            for n in range(1, self.degree):
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)
                q_np1 = q_n * q
                q_nm1 = q_n / (q + 1e-12)

                # Coefficients
                A_coeff = 2.0 * (1.0 - beta * q_n)  # Multiplies x * C_n
                C_coeff = 1.0 - beta * beta * q_nm1  # Multiplies C_{n-1}
                denom = 1.0 - q_np1

                C_next = (A_coeff * x * C_curr - C_coeff * C_prev) / (denom + 1e-12)
                basis.append(C_next)
                C_prev = C_curr
                C_curr = C_next

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "beta": self.beta_init,
            "beta_trainable": self.beta_trainable,
        })
        return config


# =============================================================================
# Sprint 7H: Final q-Polynomials
# =============================================================================


@tfk.utils.register_keras_serializable(package="arnold", name="QuantumQKrawtchouk")
class QuantumQKrawtchouk(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Quantum q-Krawtchouk polynomials.

    The Quantum q-Krawtchouk polynomials :math:`K_n^{qtm}(x; p, N | q)` are
    a quantum-group-theoretic variant of q-Krawtchouk polynomials. They arise 
    in the representation theory of the quantum group :math:`U_q(su(2))`.

    They satisfy a three-term recurrence:

    .. math::

        \lambda(x) K_n(x) = A_n K_{n+1}(x) + B_n K_n(x) + C_n K_{n-1}(x)

    where the spectral variable :math:`\lambda(x)` and coefficients depend on 
    the quantum deformation parameter q.

    Parameters
    ----------
    p : float
        Parameter p > 0. Default is 0.5.
    N : int
        Discrete support size, must be >= degree.
    p_trainable : bool
        Whether p is trainable. Default is False.
    q : float
        Base parameter in (0, 1). Default is 0.5.

    Notes
    -----
    - Related to Clebsch-Gordan coefficients for :math:`U_q(su(2))`
    - In limit q → 1, reduces to classical Krawtchouk
    - Has applications in quantum computing and quantum information theory

    References
    ----------
    .. [1] Koekoek et al. (2010), Chapter 14.15 - Quantum q-Krawtchouk
    .. [2] Koornwinder (1989). "Krawtchouk polynomials, a unification of two different group theoretic interpretations"
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        p: float = 0.5,
        N: int = 10,
        p_trainable: bool = False,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        if N < degree:
            raise ValueError(f"N must be >= degree, got N={N}, degree={degree}")

        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )

        self.p_init = p
        self.N = N
        self.p_trainable = p_trainable
        self.p_logits = None

    def build(self, input_shape):
        super().build(input_shape)

        p_logit = _inverse_softplus_lower_bound(self.p_init, 0.0)

        self.p_logits = self.add_weight(
            name="p_logits",
            shape=(),
            initializer=tf.constant_initializer(p_logit),
            trainable=self.p_trainable,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Compute Quantum q-Krawtchouk polynomial basis using three-term recurrence.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, degree+1).
        """
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)

        q = self._get_q(tf.float64)
        p = softplus_lower_bound(tf.cast(self.p_logits, tf.float64), 0.0)
        N = tf.cast(self.N, tf.float64)

        ones = tf.ones_like(x, dtype=tf.float64)

        # K_0(x) = 1
        K0 = ones
        basis = [K0]

        if self.degree >= 1:
            # Three-term recurrence for Quantum q-Krawtchouk
            # Based on Koekoek et al. (2010)

            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for Quantum q-Krawtchouk."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)
                q_N = tf.pow(q, N)

                # A_n coefficient (of K_{n+1})
                A_n = (1.0 - q_n * q) * (1.0 - p * q_n * q)

                # C_n coefficient (of K_{n-1})
                C_n = p * q * (1.0 - q_n) * (q_N - q_n)

                # B_n = spectral normalization
                B_n = p * q + q_N - A_n - C_n

                return A_n, B_n, C_n

            # K_1(x)
            A0, B0, _ = compute_coeffs(0)
            K1 = (x - B0 * ones) / (A0 + 1e-12)
            basis.append(K1)

            # K_n for n >= 2
            K_prev = K0
            K_curr = K1

            for n in range(1, self.degree):
                A_n, B_n, C_n = compute_coeffs(n)
                K_next = (x * K_curr - B_n * K_curr - C_n * K_prev) / (A_n + 1e-12)
                basis.append(K_next)
                K_prev = K_curr
                K_curr = K_next

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "p": self.p_init,
            "N": self.N,
            "p_trainable": self.p_trainable,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="ContinuousQLaguerre")
class ContinuousQLaguerre(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Continuous q-Laguerre polynomials.

    The Continuous q-Laguerre polynomials :math:`P_n^{(\alpha)}(x | q)` are 
    q-analogs of the classical Laguerre polynomials, orthogonal on :math:`[0, \infty)`.

    They satisfy a three-term recurrence:

    .. math::

        x P_n(x) = A_n P_{n+1}(x) + B_n P_n(x) + C_n P_{n-1}(x)

    where the coefficients depend on α and q.

    Parameters
    ----------
    alpha : float
        Parameter α > -1. Default is 0.0.
    alpha_trainable : bool
        Whether α is trainable. Default is False.
    q : float
        Base parameter in (0, 1). Default is 0.5.

    Notes
    -----
    - Limit: :math:`\lim_{q \to 1} P_n^{(\alpha)}(x(1-q) | q) = L_n^{(\alpha)}(x)`
    - Related to little q-Jacobi polynomials via limiting case
    - Applications in q-harmonic analysis and quantum mechanics

    References
    ----------
    .. [1] Koekoek et al. (2010), Chapter 14.21 - Continuous q-Laguerre
    .. [2] Moak (1981). "The q-analogue of the Laguerre polynomials"
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        alpha: float = 0.0,
        alpha_trainable: bool = False,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = (0.0, 2.0),
        **kwargs,
    ):
        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )

        self.alpha_init = alpha
        self.alpha_trainable = alpha_trainable
        self.alpha_logits = None

    def build(self, input_shape):
        super().build(input_shape)

        # α > -1 via softplus with lower bound
        alpha_logit = _inverse_softplus_lower_bound(self.alpha_init + 1.0, 0.0)

        self.alpha_logits = self.add_weight(
            name="alpha_logits",
            shape=(),
            initializer=tf.constant_initializer(alpha_logit),
            trainable=self.alpha_trainable,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Compute Continuous q-Laguerre polynomial basis using three-term recurrence.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, degree+1).
        """
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)

        q = self._get_q(tf.float64)
        alpha = softplus_lower_bound(tf.cast(self.alpha_logits, tf.float64), 0.0) - 1.0

        # P_0(x) = 1
        P0 = tf.ones_like(x, dtype=tf.float64)
        basis = [P0]

        if self.degree >= 1:
            # Three-term recurrence for Continuous q-Laguerre
            q_a = tf.pow(q, alpha)

            def compute_coeffs(n):
                """Compute A_n, B_n, C_n for Continuous q-Laguerre."""
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)

                # A_n (coefficient of P_{n+1})
                A_n = -q_n * q / (1.0 - q_a * q_n * q * q)

                # C_n (coefficient of P_{n-1})
                C_n = q_a * q_n * (1.0 - q_n) / (1.0 - q_a * q_n * q)

                # B_n
                B_n = 1.0 + q_a * q_n * q - A_n * (1.0 - q_a * q_n * q * q) - C_n

                return A_n, B_n, C_n

            # P_1(x)
            A0, B0, _ = compute_coeffs(0)
            P1 = (x - B0) / (A0 + 1e-12)
            basis.append(P1)

            # P_n for n >= 2
            P_prev = P0
            P_curr = P1

            for n in range(1, self.degree):
                A_n, B_n, C_n = compute_coeffs(n)
                P_next = (x * P_curr - B_n * P_curr - C_n * P_prev) / (A_n + 1e-12)
                basis.append(P_next)
                P_prev = P_curr
                P_curr = P_next

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha_init,
            "alpha_trainable": self.alpha_trainable,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="ContinuousQLegendre")
class ContinuousQLegendre(QPolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Continuous q-Legendre polynomials.

    The Continuous q-Legendre polynomials :math:`P_n(x | q)` are the special case
    of continuous q-ultraspherical polynomials with :math:`\beta = q^{1/2}`:

    .. math::

        P_n(x | q) = C_n(x; q^{1/2} | q)

    They satisfy a three-term recurrence:

    .. math::

        2x(1 - q^{n+1/2}) P_n(x) = (1 - q^{n+1}) P_{n+1}(x) + (1 - q^n) P_{n-1}(x)

    with :math:`P_0(x | q) = 1` and :math:`P_1(x | q) = 2x(1-q^{1/2})/(1-q)`.

    Parameters
    ----------
    q : float
        Base parameter in (0, 1). Default is 0.5.
    q_trainable : bool
        Whether q is trainable. Default is True.

    Notes
    -----
    - Special case: β = q^{1/2} of continuous q-ultraspherical
    - Limit: :math:`\lim_{q \to 1} P_n(x | q) = P_n(x)` (Legendre)
    - Simpler coefficients than general q-ultraspherical
    - Symmetric: :math:`P_n(-x | q) = (-1)^n P_n(x | q)`

    References
    ----------
    .. [1] NIST DLMF 18.28(v) - Special case of Continuous q-Ultraspherical
    .. [2] Koekoek et al. (2010), Chapter 14.10.1
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        q: float = 0.5,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        **kwargs,
    ):
        super().__init__(
            degree=degree, units=units, q=q, q_trainable=q_trainable,
            input_clip=input_clip, **kwargs
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Compute Continuous q-Legendre polynomial basis.

        Uses the three-term recurrence with β = q^{1/2}:

        .. math::

            2x(1 - q^{n+1/2}) P_n = (1 - q^{n+1}) P_{n+1} + (1 - q^n) P_{n-1}

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, degree+1).
        """
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)

        q = self._get_q(tf.float64)
        sqrt_q = tf.sqrt(q)

        # P_0(x) = 1
        P0 = tf.ones_like(x, dtype=tf.float64)
        basis = [P0]

        if self.degree >= 1:
            # P_1(x) = 2x(1 - sqrt(q)) / (1 - q)
            P1 = 2.0 * x * (1.0 - sqrt_q) / (1.0 - q + 1e-12)
            basis.append(P1)

            # Recurrence: 2x(1 - q^{n+1/2}) P_n = (1 - q^{n+1}) P_{n+1} + (1 - q^n) P_{n-1}
            # Rearranged: P_{n+1} = [2x(1 - q^{n+1/2}) P_n - (1 - q^n) P_{n-1}] / (1 - q^{n+1})
            P_prev = P0
            P_curr = P1

            for n in range(1, self.degree):
                n_f = tf.cast(n, tf.float64)
                q_n = tf.pow(q, n_f)
                q_np1 = q_n * q
                q_n_half = q_n * sqrt_q  # q^{n+1/2}

                # Coefficients
                A_coeff = 2.0 * (1.0 - q_n_half)  # Multiplies x * P_n
                C_coeff = 1.0 - q_n  # Multiplies P_{n-1}
                denom = 1.0 - q_np1

                P_next = (A_coeff * x * P_curr - C_coeff * P_prev) / (denom + 1e-12)
                basis.append(P_next)
                P_prev = P_curr
                P_curr = P_next

        result = tf.stack(basis, axis=-1)
        return tf.cast(result, orig_dtype)
