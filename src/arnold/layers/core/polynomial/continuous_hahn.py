## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""
Continuous Hahn Family Polynomial KAN layers.

This module provides KAN layers using continuous Hahn family polynomial bases:
- ContinuousHahn: Generalizes Jacobi with complex parameters, orthogonal on (-∞, ∞)
- ContinuousDualHahn: Dual of ContinuousHahn, part of Wilson class
- DualHahn: Discrete dual of Hahn polynomials
- Stieltjes: Moment problems, measure theory

All polynomials use three-term recurrence for numerical stability.
"""
import math

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


@tfk.utils.register_keras_serializable(package="arnold", name="ContinuousHahn")
class ContinuousHahn(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Continuous Hahn polynomials.

    The Continuous Hahn polynomials :math:`p_n(x; a, b, c, d)` are orthogonal
    on :math:`(-\infty, \infty)` with weight function:

    .. math::

        w(x) = |\Gamma(a + ix) \Gamma(b + ix) \Gamma(c - ix) \Gamma(d - ix)|^2

    They are defined as:

    .. math::

        p_n(x; a, b, c, d) = i^n \frac{(a+c)_n (a+d)_n}{n!} 
            \,_3F_2\left(\begin{array}{c} -n, n+a+b+c+d-1, a+ix \\ a+c, a+d \end{array}; 1\right)

    The three-term recurrence (monic form) is:

    .. math::

        x \tilde{p}_n(x) = \tilde{p}_{n+1}(x) + i(A_n + C_n) \tilde{p}_n(x) 
            - A_{n-1} C_n \tilde{p}_{n-1}(x)

    where:

    .. math::

        A_n &= -\frac{(n + a + b + c + d - 1)(n + a + c)(n + a + d)}
                    {(2n + a + b + c + d - 1)(2n + a + b + c + d)} \\
        C_n &= \frac{n(n + b + c - 1)(n + b + d - 1)}
                    {(2n + a + b + c + d - 2)(2n + a + b + c + d - 1)}

    Parameters
    ----------
    a, b : float
        Parameters with Re(a) > 0, Re(b) > 0. In this implementation, 
        we use real parameters with a, b > 0.
    c, d : float | None
        Parameters. For orthogonality on the real line, we need c = conj(a), d = conj(b).
        If None, defaults to c = a, d = b (real case).
    trainable_params : bool
        Whether parameters are learnable.

    Notes
    -----
    Continuous Hahn polynomials generalize Jacobi polynomials and are in the
    Hahn class of the Askey scheme. They have applications in signal processing
    and representation theory.

    References
    ----------
    .. [1] NIST DLMF 18.19 - Hahn Class: Definitions
    .. [2] Koekoek, R., Lesky, P. A., & Swarttouw, R. F. (2010). 
           Hypergeometric orthogonal polynomials and their q-analogues.
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        a: float = 1.0,
        b: float = 1.0,
        c: float | None = None,
        d: float | None = None,
        trainable_params: bool = False,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)
        
        self.a_init = a
        self.b_init = b
        self.c_init = c if c is not None else a
        self.d_init = d if d is not None else b
        self.trainable_params = trainable_params
        
        # Logit variables for softplus constraints
        self.a_logits = None
        self.b_logits = None
        self.c_logits = None
        self.d_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        
        # Parameters must be > 0 for orthogonality
        # Use softplus to enforce positivity
        a_logit = _inverse_softplus_lower_bound(self.a_init, 0.0)
        b_logit = _inverse_softplus_lower_bound(self.b_init, 0.0)
        c_logit = _inverse_softplus_lower_bound(self.c_init, 0.0)
        d_logit = _inverse_softplus_lower_bound(self.d_init, 0.0)
        
        self.a_logits = self.add_weight(
            name="a_logits",
            shape=(),
            initializer=tf.constant_initializer(a_logit),
            trainable=self.trainable_params,
            dtype=self.dtype,
        )
        self.b_logits = self.add_weight(
            name="b_logits",
            shape=(),
            initializer=tf.constant_initializer(b_logit),
            trainable=self.trainable_params,
            dtype=self.dtype,
        )
        self.c_logits = self.add_weight(
            name="c_logits",
            shape=(),
            initializer=tf.constant_initializer(c_logit),
            trainable=self.trainable_params,
            dtype=self.dtype,
        )
        self.d_logits = self.add_weight(
            name="d_logits",
            shape=(),
            initializer=tf.constant_initializer(d_logit),
            trainable=self.trainable_params,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        """Compute Continuous Hahn polynomial basis."""
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        
        # Get positive parameters via softplus
        a = softplus_lower_bound(tf.cast(self.a_logits, tf.float64), 0.0)
        b = softplus_lower_bound(tf.cast(self.b_logits, tf.float64), 0.0)
        c = softplus_lower_bound(tf.cast(self.c_logits, tf.float64), 0.0)
        d = softplus_lower_bound(tf.cast(self.d_logits, tf.float64), 0.0)
        
        s = a + b + c + d  # Sum of parameters
        
        # Use monic recurrence: x * p̃_n = p̃_{n+1} + i(A_n + C_n) p̃_n - A_{n-1} C_n p̃_{n-1}
        # Since we work with real x, we use a modified recurrence avoiding complex arithmetic.
        # 
        # For real parameters (c=a, d=b), we can use:
        # The recurrence becomes real when parameters satisfy the conjugate condition.
        # We implement the real-valued version.
        
        ones = tf.ones_like(x, dtype=tf.float64)
        
        # p_0 = 1
        P0 = ones
        
        basis = [P0]
        
        if self.degree >= 1:
            # p_1(x) = x - i(A_0 + C_0) where C_0 = 0
            # A_0 = -(s-1) * (a+c) * (a+d) / ((s-1) * s)
            #     = -(a+c)(a+d) / s
            A0 = -(a + c) * (a + d) / s
            # For real case with c=a, d=b: A_0 = -(a+a)(a+b)/s = -2a(a+b)/s
            # C_0 = 0 (n=0 term)
            
            # p_1(x) involves imaginary shift, but for evaluation we use:
            # In the real parameter case, the polynomial evaluates to real values
            # We compute using the standard form
            P1 = x - A0  # First order term (simplified for real case)
            basis.append(P1)
        
        P_prev2 = P0
        P_prev1 = P1 if self.degree >= 1 else P0
        
        for n in range(1, self.degree):
            n_f = tf.constant(float(n), dtype=tf.float64)
            n1_f = tf.constant(float(n + 1), dtype=tf.float64)
            
            # Recurrence coefficients
            # A_n = -(n + s - 1)(n + a + c)(n + a + d) / ((2n + s - 1)(2n + s))
            An = -(n_f + s - 1.0) * (n_f + a + c) * (n_f + a + d) / (
                (2.0 * n_f + s - 1.0) * (2.0 * n_f + s)
            )
            
            # C_n = n(n + b + c - 1)(n + b + d - 1) / ((2n + s - 2)(2n + s - 1))
            Cn = n_f * (n_f + b + c - 1.0) * (n_f + b + d - 1.0) / (
                (2.0 * n_f + s - 2.0) * (2.0 * n_f + s - 1.0)
            )
            
            # A_{n-1} for the recurrence
            n_prev = n_f - 1.0
            An_prev = -(n_prev + s - 1.0) * (n_prev + a + c) * (n_prev + a + d) / (
                (2.0 * n_prev + s - 1.0) * (2.0 * n_prev + s)
            ) if n > 0 else tf.constant(0.0, dtype=tf.float64)
            
            # Monic recurrence: p_{n+1} = (x - (A_n + C_n)) p_n - A_{n-1} C_n p_{n-1}
            # The shift term (A_n + C_n) is real for real symmetric parameters
            shift = An + Cn
            coeff = An_prev * Cn
            
            P_n = (x - shift) * P_prev1 - coeff * P_prev2
            basis.append(P_n)
            
            P_prev2 = P_prev1
            P_prev1 = P_n
        
        basis_tensor = tf.stack(basis, axis=-1)
        return tf.cast(basis_tensor, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "a": self.a_init,
            "b": self.b_init,
            "c": self.c_init,
            "d": self.d_init,
            "trainable_params": self.trainable_params,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="ContinuousDualHahn")
class ContinuousDualHahn(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Continuous Dual Hahn polynomials.

    The Continuous Dual Hahn polynomials :math:`S_n(x^2; a, b, c)` are part of
    the Wilson class in the Askey scheme. They are defined as:

    .. math::

        S_n(x^2; a, b, c) = \,_3F_2\left(\begin{array}{c} -n, a+ix, a-ix \\ 
            a+b, a+c \end{array}; 1\right)

    They are orthogonal on :math:`(0, \infty)` with weight function:

    .. math::

        w(y^2) = \frac{1}{2y} \left|\frac{\prod_j \Gamma(a_j + iy)}{\Gamma(2iy)}\right|^2

    The three-term recurrence uses :math:`t = x^2`:

    .. math::

        S_0 &= 1 \\
        S_1 &= t - (A_0 - a^2) \\
        S_{n+1} &= (t - (A_n + C_n - a^2)) S_n - A_{n-1} C_n S_{n-1}

    where (from Wilson polynomials with d→∞):

    .. math::

        A_n &= (n + a + b)(n + a + c) \\
        C_n &= n(n + b + c - 1)

    Parameters
    ----------
    a, b, c : float
        Parameters with Re(a), Re(b), Re(c) > 0 or nonreal in conjugate pairs.
    trainable_params : bool
        Whether parameters are learnable.

    Notes
    -----
    Continuous Dual Hahn polynomials are obtained as a limit of Wilson polynomials
    when one parameter goes to infinity. They have the generating function:

    .. math::

        (1-z)^{-c+iy} \,_2F_1\left(\begin{array}{c} a+iy, b+iy \\ 
            a+b \end{array}; z\right) = \sum_{n=0}^{\infty} \frac{S_n(y^2; a, b, c)}{(a+b)_n n!} z^n

    References
    ----------
    .. [1] NIST DLMF 18.25 - Wilson Class: Definitions
    .. [2] NIST DLMF 18.26 - Wilson Class: Continued
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        a: float = 1.0,
        b: float = 1.0,
        c: float = 1.0,
        trainable_params: bool = False,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)
        
        self.a_init = a
        self.b_init = b
        self.c_init = c
        self.trainable_params = trainable_params
        
        self.a_logits = None
        self.b_logits = None
        self.c_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        
        a_logit = _inverse_softplus_lower_bound(self.a_init, 0.0)
        b_logit = _inverse_softplus_lower_bound(self.b_init, 0.0)
        c_logit = _inverse_softplus_lower_bound(self.c_init, 0.0)
        
        self.a_logits = self.add_weight(
            name="a_logits",
            shape=(),
            initializer=tf.constant_initializer(a_logit),
            trainable=self.trainable_params,
            dtype=self.dtype,
        )
        self.b_logits = self.add_weight(
            name="b_logits",
            shape=(),
            initializer=tf.constant_initializer(b_logit),
            trainable=self.trainable_params,
            dtype=self.dtype,
        )
        self.c_logits = self.add_weight(
            name="c_logits",
            shape=(),
            initializer=tf.constant_initializer(c_logit),
            trainable=self.trainable_params,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        """Compute Continuous Dual Hahn polynomial basis."""
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        
        a = softplus_lower_bound(tf.cast(self.a_logits, tf.float64), 0.0)
        b = softplus_lower_bound(tf.cast(self.b_logits, tf.float64), 0.0)
        c = softplus_lower_bound(tf.cast(self.c_logits, tf.float64), 0.0)
        
        # The polynomial is in x^2, so we use t = x^2
        t = tf.square(x)
        ones = tf.ones_like(t, dtype=tf.float64)
        
        def A(n_f):
            return (n_f + a + b) * (n_f + a + c)
        
        def C(n_f):
            return n_f * (n_f + b + c - 1.0)
        
        a2 = tf.square(a)
        A0 = A(tf.constant(0.0, dtype=tf.float64))
        
        # S_0 = 1
        S0 = ones
        basis = [S0]
        
        if self.degree >= 1:
            # S_1 = t - (A_0 - a^2) = t - (a+b)(a+c) + a^2
            S1 = t - (A0 - a2)
            basis.append(S1)
        
        S_prev2 = S0
        S_prev1 = S1 if self.degree >= 1 else S0
        
        for n in range(1, self.degree):
            n_f = tf.constant(float(n), dtype=tf.float64)
            
            An = A(n_f)
            Cn = C(n_f)
            An_prev = A(n_f - 1.0) if n > 0 else tf.constant(0.0, dtype=tf.float64)
            
            # S_{n+1} = (t - (A_n + C_n - a^2)) S_n - A_{n-1} C_n S_{n-1}
            shift = An + Cn - a2
            coeff = An_prev * Cn
            
            S_n = (t - shift) * S_prev1 - coeff * S_prev2
            basis.append(S_n)
            
            S_prev2 = S_prev1
            S_prev1 = S_n
        
        basis_tensor = tf.stack(basis, axis=-1)
        return tf.cast(basis_tensor, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "a": self.a_init,
            "b": self.b_init,
            "c": self.c_init,
            "trainable_params": self.trainable_params,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="DualHahn")
class DualHahn(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Dual Hahn polynomials.

    The Dual Hahn polynomials :math:`R_n(\lambda(x); \gamma, \delta, N)` are
    discrete orthogonal polynomials on :math:`\{0, 1, \ldots, N\}`.

    They are defined via the duality relation with Hahn polynomials:

    .. math::

        R_n(y(y + \gamma + \delta + 1); \gamma, \delta, N) = Q_y(n; \gamma, \delta, N)

    where :math:`Q_n` are Hahn polynomials and :math:`\lambda(y) = y(y + \gamma + \delta + 1)`.

    The orthogonality is:

    .. math::

        \sum_{y=0}^{N} R_m(\lambda(y)) R_n(\lambda(y)) \frac{\gamma + \delta + 1 + 2y}
            {\gamma + \delta + 1 + y} \omega_y = h_n \delta_{mn}

    The three-term recurrence in :math:`\lambda = x(x + \gamma + \delta + 1)`:

    .. math::

        R_0 &= 1 \\
        R_1 &= 1 - \frac{\lambda}{A_0} \\
        R_{n+1} &= \frac{(A_n + C_n - \lambda) R_n - C_n R_{n-1}}{A_n}

    where:

    .. math::

        A_n &= (n + \gamma + 1)(n - N) \\
        C_n &= n(n - \delta - N - 1)

    Parameters
    ----------
    gamma : float
        Shape parameter with γ > -1 or γ < -N.
    delta : float
        Shape parameter with δ > -1 or δ < -N.
    N : int
        Size of discrete domain {0, 1, ..., N}. Must be ≥ degree.
    trainable_params : bool
        Whether parameters are learnable.

    Notes
    -----
    Dual Hahn polynomials are the discrete analog of Continuous Dual Hahn and
    form the dual of Hahn polynomials. They appear in combinatorics and
    representation theory.

    References
    ----------
    .. [1] NIST DLMF 18.25 - Wilson Class: Definitions
    .. [2] Koekoek, R., Lesky, P. A., & Swarttouw, R. F. (2010).
           Hypergeometric orthogonal polynomials and their q-analogues.
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        gamma: float = 1.0,
        delta: float = 1.0,
        N: int = 10,
        trainable_params: bool = False,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        if N < degree:
            raise ValueError(f"N={N} must be >= degree={degree}")
        
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)
        
        self.gamma_init = gamma
        self.delta_init = delta
        self.N = N
        self.trainable_params = trainable_params
        
        self.gamma_logits = None
        self.delta_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        
        # gamma, delta > -1 for standard orthogonality
        gamma_logit = _inverse_softplus_lower_bound(self.gamma_init, -1.0)
        delta_logit = _inverse_softplus_lower_bound(self.delta_init, -1.0)
        
        self.gamma_logits = self.add_weight(
            name="gamma_logits",
            shape=(),
            initializer=tf.constant_initializer(gamma_logit),
            trainable=self.trainable_params,
            dtype=self.dtype,
        )
        self.delta_logits = self.add_weight(
            name="delta_logits",
            shape=(),
            initializer=tf.constant_initializer(delta_logit),
            trainable=self.trainable_params,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        """Compute Dual Hahn polynomial basis."""
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        
        gamma = softplus_lower_bound(tf.cast(self.gamma_logits, tf.float64), -1.0)
        delta = softplus_lower_bound(tf.cast(self.delta_logits, tf.float64), -1.0)
        N_f = tf.constant(float(self.N), dtype=tf.float64)
        
        # λ(x) = x(x + γ + δ + 1)
        gd = gamma + delta + 1.0
        lam = x * (x + gd)
        
        ones = tf.ones_like(x, dtype=tf.float64)
        
        def A(n_f):
            return (n_f + gamma + 1.0) * (n_f - N_f)
        
        def C(n_f):
            return n_f * (n_f - delta - N_f - 1.0)
        
        A0 = A(tf.constant(0.0, dtype=tf.float64))
        
        # R_0 = 1
        R0 = ones
        basis = [R0]
        
        if self.degree >= 1:
            # R_1 = 1 - λ / A_0
            R1 = ones - lam / A0
            basis.append(R1)
        
        R_prev2 = R0
        R_prev1 = R1 if self.degree >= 1 else R0
        
        for n in range(1, self.degree):
            n_f = tf.constant(float(n), dtype=tf.float64)
            
            An = A(n_f)
            Cn = C(n_f)
            
            # R_{n+1} = ((A_n + C_n - λ) R_n - C_n R_{n-1}) / A_n
            R_n = ((An + Cn - lam) * R_prev1 - Cn * R_prev2) / An
            basis.append(R_n)
            
            R_prev2 = R_prev1
            R_prev1 = R_n
        
        basis_tensor = tf.stack(basis, axis=-1)
        return tf.cast(basis_tensor, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma_init,
            "delta": self.delta_init,
            "N": self.N,
            "trainable_params": self.trainable_params,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="StieltjesWigert")
class StieltjesWigert(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Stieltjes-Wigert polynomials.

    The Stieltjes-Wigert polynomials :math:`S_n(x; q)` are q-orthogonal polynomials
    related to the log-normal distribution. They are defined as:

    .. math::

        S_n(x; q) = \frac{1}{(q; q)_n} \,_1\phi_1\left(\begin{array}{c} q^{-n} \\ 
            0 \end{array}; q, -q^{n+1} x\right)

    They are orthogonal with respect to the weight function:

    .. math::

        w(x) = \frac{1}{(-x, -qx^{-1}; q)_\infty}

    on :math:`(0, \infty)`, or equivalently with respect to the log-normal weight:

    .. math::

        w(x) = \exp\left(-\frac{(\ln x)^2}{2 \ln(q^{-1})}\right)

    The three-term recurrence is:

    .. math::

        S_0 &= 1 \\
        S_1 &= 1 - (q; q)_1 x \\
        (q; q)_{n+1} x S_n &= S_{n+1} - (1 + q^n - (q; q)_n q^n x) S_n + q^n S_{n-1}

    Parameters
    ----------
    q : float
        Base parameter with 0 < q < 1.
    trainable_params : bool
        Whether q is learnable.

    Notes
    -----
    Stieltjes-Wigert polynomials arise in the study of moment problems
    (specifically indeterminate moment problems) and have connections to
    log-normal distributions and quantum groups.

    As q → 1, they converge to Hermite polynomials:

    .. math::

        \lim_{q \to 1} (q; q)_n S_n(q^{-1} x \sqrt{2(1-q)} + 1; q) / (1-q^2)^{n/2} 
            = (-1)^n H_n(x)

    References
    ----------
    .. [1] NIST DLMF 18.27.18-20 - Stieltjes-Wigert Polynomials
    .. [2] Stieltjes, T. J. (1894). Recherches sur les fractions continues.
           Annales de la Faculté des sciences de Toulouse.
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        q: float = 0.5,
        trainable_params: bool = False,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        if not 0 < q < 1:
            raise ValueError(f"q must be in (0, 1), got {q}")
        
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)
        
        self.q_init = q
        self.trainable_params = trainable_params
        
        self.q_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        
        # q ∈ (0, 1) via sigmoid
        q_logit = math.log(self.q_init / (1.0 - self.q_init))  # inverse sigmoid
        
        self.q_logits = self.add_weight(
            name="q_logits",
            shape=(),
            initializer=tf.constant_initializer(q_logit),
            trainable=self.trainable_params,
            dtype=self.dtype,
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        """Compute Stieltjes-Wigert polynomial basis."""
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float64)
        
        # q ∈ (0, 1) via sigmoid
        q = tf.nn.sigmoid(tf.cast(self.q_logits, tf.float64))
        # Clamp away from boundaries for numerical stability
        q = tf.clip_by_value(q, 1e-6, 1.0 - 1e-6)
        
        ones = tf.ones_like(x, dtype=tf.float64)
        
        # Precompute q-Pochhammer symbols (q; q)_n = prod_{k=0}^{n-1} (1 - q^k)
        q_poch = [tf.constant(1.0, dtype=tf.float64)]  # (q;q)_0 = 1
        prod = tf.constant(1.0, dtype=tf.float64)
        for k in range(1, self.degree + 2):
            prod = prod * (1.0 - tf.pow(q, tf.constant(float(k - 1), dtype=tf.float64)))
            q_poch.append(prod)
        
        # S_0 = 1
        S0 = ones
        basis = [S0]
        
        if self.degree >= 1:
            # S_1 = 1 - (q;q)_1 * x = 1 - (1-q) * x
            S1 = ones - q_poch[1] * x
            basis.append(S1)
        
        S_prev2 = S0
        S_prev1 = S1 if self.degree >= 1 else S0
        
        for n in range(1, self.degree):
            n_f = tf.constant(float(n), dtype=tf.float64)
            qn = tf.pow(q, n_f)
            
            # Recurrence: (q;q)_{n+1} x S_n = S_{n+1} - (1 + q^n - (q;q)_n q^n x) S_n + q^n S_{n-1}
            # Rearranging: S_{n+1} = (q;q)_{n+1} x S_n + (1 + q^n - (q;q)_n q^n x) S_n - q^n S_{n-1}
            #                      = ((q;q)_{n+1} - (q;q)_n q^n) x S_n + (1 + q^n) S_n - q^n S_{n-1}
            
            coeff_x = q_poch[n + 1] - q_poch[n] * qn
            coeff_const = 1.0 + qn
            
            S_n = coeff_x * x * S_prev1 + coeff_const * S_prev1 - qn * S_prev2
            basis.append(S_n)
            
            S_prev2 = S_prev1
            S_prev1 = S_n
        
        basis_tensor = tf.stack(basis, axis=-1)
        return tf.cast(basis_tensor, orig_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "q": self.q_init,
            "trainable_params": self.trainable_params,
        })
        return config
