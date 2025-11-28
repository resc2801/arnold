## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""
Discrete Orthogonal Polynomial KAN layers.

This module provides KAN layers using discrete orthogonal polynomial bases:
- Krawtchouk: Image moments, pattern recognition
- Hahn: Combinatorics, statistics (generalizes Krawtchouk)
- Meixner: Stochastic processes, queuing theory
- Racah: Quantum mechanics, 6j-symbols (most general)

All polynomials use three-term recurrence for numerical stability
and log-gamma computations for parameter constraints.
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
    softplus_lower_bound(x) = softplus(x) + lower_bound + eps
    So we need: x = inverse_softplus(value - lower_bound - eps)
    where inverse_softplus(y) = log(exp(y) - 1)
    """
    y = value - lower_bound - eps
    if y <= 0:
        raise ValueError(f"value={value} must be > lower_bound + eps = {lower_bound + eps}")
    # inverse_softplus(y) = log(exp(y) - 1)
    # For numerical stability, when y is large: log(exp(y) - 1) ≈ y
    if y > 20:
        return y
    return math.log(math.exp(y) - 1)


@tfk.utils.register_keras_serializable(package="arnold", name="Krawtchouk")
class Krawtchouk(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Krawtchouk polynomials.

    The Krawtchouk polynomials :math:`K_n(x; p, N)` are discrete orthogonal
    polynomials on :math:`\{0, 1, \ldots, N\}` with binomial weight function:

    .. math::

        w(x) = \binom{N}{x} p^x (1-p)^{N-x}

    They satisfy the three-term recurrence:

    .. math::

        K_0(x) &= 1 \\
        K_1(x) &= 1 - \frac{x}{Np} \\
        (n+1) K_{n+1}(x) &= [(N-2x)p + (n+1) - Np(1-p)] K_n(x) 
                          - (N-n+1)(1-p) K_{n-1}(x)

    Alternative form using DLMF 18.22.2:

    .. math::

        -x K_n(x) = A_n K_{n+1}(x) - (A_n + C_n) K_n(x) + C_n K_{n-1}(x)

    where:

    .. math::

        A_n &= p(N-n) \\
        C_n &= n(1-p)

    Parameters
    ----------
    p_init : float | None
        Initial value for probability parameter p ∈ (0, 1). Default 0.5.
    p_trainable : bool
        Whether p is learnable.
    N : int
        Size of discrete domain {0, 1, ..., N}. Must be ≥ degree.

    Notes
    -----
    Krawtchouk polynomials are widely used in image processing for computing
    discrete orthogonal moments, providing shift, scale, and rotation invariant
    features [1]_.

    References
    ----------
    .. [1] Yap, P. T., Paramesran, R., & Ong, S. H. (2003). Image analysis by
           Krawtchouk moments. IEEE TIP, 12(11), 1367-1377.

    See Also
    --------
    Hahn : Generalization with two shape parameters
    Meixner : Different discrete weight function
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        p_init: float = 0.5,
        p_trainable: bool = True,
        N: int = 10,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        degree : int
            Maximum polynomial degree. Must be ≤ N.
        units : int
            Output dimensionality.
        p_init : float
            Initial probability parameter (must be in (0, 1)).
        p_trainable : bool
            Whether p is trainable.
        N : int
            Size of discrete support. Must be ≥ degree.
        input_clip : tuple[float, float] | None
            Optional input clamp.
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        if N < degree:
            raise ValueError(f"N must be >= degree, got N={N}, degree={degree}")
        if not 0 < p_init < 1:
            raise ValueError(f"p_init must be in (0, 1), got {p_init}")
        
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)
        
        self.p_init = p_init
        self.p_trainable = p_trainable
        self.N = N
        self._p_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        # Map p_init to logits via inverse sigmoid: logit = log(p / (1-p))
        # Use Python math.log for graph-mode compatibility
        p_logit_init = math.log(self.p_init / (1.0 - self.p_init))
        self._p_logits = self.add_weight(
            shape=(),
            initializer=tfk.initializers.Constant(p_logit_init),
            name="p_logits",
            trainable=self.p_trainable,
        )

    @kan_fn
    def pseudo_vandermonde(self, x: tf.Tensor) -> tf.Tensor:
        """Compute Krawtchouk basis using three-term recurrence."""
        # p ∈ (0, 1) via sigmoid
        p = tf.sigmoid(tf.cast(self._p_logits, x.dtype))
        N = tf.cast(self.N, x.dtype)
        
        # K_0(x) = 1
        basis = [tf.ones_like(x)]
        
        if self.degree > 0:
            # K_1(x) = 1 - x/(Np)
            K1 = 1.0 - x / (N * p + 1e-8)
            basis.append(K1)
        
        # Use DLMF recurrence: -x K_n = A_n K_{n+1} - (A_n + C_n) K_n + C_n K_{n-1}
        # Rearranged: K_{n+1} = [(A_n + C_n - x) K_n - C_n K_{n-1}] / A_n
        for n in range(1, self.degree):
            n_f = tf.cast(n, x.dtype)
            A_n = p * (N - n_f)
            C_n = n_f * (1.0 - p)
            # Avoid division by zero when n approaches N
            A_n_safe = A_n + 1e-8
            K_next = ((A_n + C_n - x) * basis[n] - C_n * basis[n - 1]) / A_n_safe
            basis.append(K_next)
        
        return tf.stack(basis, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "p_init": self.p_init,
            "p_trainable": self.p_trainable,
            "N": self.N,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="Hahn")
class Hahn(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Hahn polynomials.

    The Hahn polynomials :math:`Q_n(x; \alpha, \beta, N)` are discrete orthogonal
    polynomials on :math:`\{0, 1, \ldots, N\}` that generalize Krawtchouk.

    Weight function:

    .. math::

        w(x) = \binom{\alpha + x}{x} \binom{\beta + N - x}{N - x}

    Three-term recurrence (DLMF 18.22.4):

    .. math::

        -x Q_n(x) = A_n Q_{n+1}(x) - (A_n + C_n) Q_n(x) + C_n Q_{n-1}(x)

    where:

    .. math::

        A_n &= \frac{(n+\alpha+\beta+1)(n+\alpha+1)(N-n)}{(2n+\alpha+\beta+1)(2n+\alpha+\beta+2)} \\
        C_n &= \frac{n(n+\alpha+\beta+N+1)(n+\beta)}{(2n+\alpha+\beta)(2n+\alpha+\beta+1)}

    Parameters
    ----------
    alpha_init, beta_init : float | None
        Initial values for shape parameters α, β > -1.
    alpha_trainable, beta_trainable : bool
        Whether parameters are learnable.
    N : int
        Size of discrete domain.

    Notes
    -----
    - When α = β = 0: Recovers Krawtchouk polynomials (up to normalization)
    - When β → ∞: Limits to Meixner polynomials
    - Important in combinatorics, coding theory, and statistics

    See Also
    --------
    Krawtchouk : Special case with simpler weight
    Racah : Further generalization (most general discrete family)
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        alpha_init: float = 0.5,
        alpha_trainable: bool = True,
        beta_init: float = 0.5,
        beta_trainable: bool = True,
        N: int = 10,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        degree : int
            Maximum polynomial degree. Must be ≤ N.
        units : int
            Output dimensionality.
        alpha_init, beta_init : float
            Initial shape parameters (must be > -1).
        alpha_trainable, beta_trainable : bool
            Whether parameters are trainable.
        N : int
            Size of discrete support.
        input_clip : tuple[float, float] | None
            Optional input clamp.
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        if N < degree:
            raise ValueError(f"N must be >= degree, got N={N}, degree={degree}")
        
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)
        
        self.alpha_init = alpha_init
        self.alpha_trainable = alpha_trainable
        self.beta_init = beta_init
        self.beta_trainable = beta_trainable
        self.N = N
        self._alpha_logits = None
        self._beta_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        # α, β > -1: use softplus with lower bound
        # Use Python inverse_softplus for graph-mode compatibility
        alpha_logit = _inverse_softplus_lower_bound(self.alpha_init, lower_bound=-1.0)
        self._alpha_logits = self.add_weight(
            shape=(),
            initializer=tfk.initializers.Constant(alpha_logit),
            name="alpha_logits",
            trainable=self.alpha_trainable,
        )
        beta_logit = _inverse_softplus_lower_bound(self.beta_init, lower_bound=-1.0)
        self._beta_logits = self.add_weight(
            shape=(),
            initializer=tfk.initializers.Constant(beta_logit),
            name="beta_logits",
            trainable=self.beta_trainable,
        )

    @kan_fn
    def pseudo_vandermonde(self, x: tf.Tensor) -> tf.Tensor:
        """Compute Hahn basis using three-term recurrence."""
        alpha = softplus_lower_bound(tf.cast(self._alpha_logits, x.dtype), lower_bound=-1.0)
        beta = softplus_lower_bound(tf.cast(self._beta_logits, x.dtype), lower_bound=-1.0)
        N = tf.cast(self.N, x.dtype)
        
        # Q_0(x) = 1
        basis = [tf.ones_like(x)]
        
        if self.degree > 0:
            # Q_1(x) from recurrence with n=0
            # A_0 = (α+β+1)(α+1)N / [(α+β+1)(α+β+2)]
            # C_0 = 0
            # -x Q_0 = A_0 Q_1 - A_0 Q_0
            # Q_1 = (A_0 - x) / A_0 = 1 - x/A_0
            s = alpha + beta
            A_0_num = (s + 1.0) * (alpha + 1.0) * N
            A_0_den = (s + 1.0) * (s + 2.0) + 1e-8
            A_0 = A_0_num / A_0_den
            Q1 = 1.0 - x / (A_0 + 1e-8)
            basis.append(Q1)
        
        for n in range(1, self.degree):
            n_f = tf.cast(n, x.dtype)
            s = alpha + beta
            
            # A_n numerator and denominator
            A_n_num = (n_f + s + 1.0) * (n_f + alpha + 1.0) * (N - n_f)
            A_n_den = (2.0 * n_f + s + 1.0) * (2.0 * n_f + s + 2.0) + 1e-8
            A_n = A_n_num / A_n_den
            
            # C_n numerator and denominator
            C_n_num = n_f * (n_f + s + N + 1.0) * (n_f + beta)
            C_n_den = (2.0 * n_f + s) * (2.0 * n_f + s + 1.0) + 1e-8
            C_n = C_n_num / C_n_den
            
            # Q_{n+1} = [(A_n + C_n - x) Q_n - C_n Q_{n-1}] / A_n
            Q_next = ((A_n + C_n - x) * basis[n] - C_n * basis[n - 1]) / (A_n + 1e-8)
            basis.append(Q_next)
        
        return tf.stack(basis, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha_init": self.alpha_init,
            "alpha_trainable": self.alpha_trainable,
            "beta_init": self.beta_init,
            "beta_trainable": self.beta_trainable,
            "N": self.N,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="Meixner")
class Meixner(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Meixner polynomials.

    The Meixner polynomials :math:`M_n(x; \beta, c)` are discrete orthogonal
    polynomials on :math:`\{0, 1, 2, \ldots\}` (unbounded support) with
    negative binomial weight:

    .. math::

        w(x) = \frac{(\beta)_x c^x}{x!}

    where :math:`(\beta)_x` is the Pochhammer symbol.

    Three-term recurrence (DLMF 18.22.9):

    .. math::

        -x M_n(x) = A_n M_{n+1}(x) - (A_n + C_n) M_n(x) + C_n M_{n-1}(x)

    where:

    .. math::

        A_n &= \frac{c(n+\beta)}{1-c} \\
        C_n &= \frac{n}{1-c}

    Parameters
    ----------
    beta_init : float | None
        Initial value for β > 0.
    beta_trainable : bool
        Whether β is learnable.
    c_init : float | None
        Initial value for c ∈ (0, 1).
    c_trainable : bool
        Whether c is learnable.

    Notes
    -----
    Meixner polynomials arise in stochastic processes, particularly:
    - Negative binomial distributions
    - Birth-death processes
    - Queuing theory

    Limiting cases:
    - As c → 0: Related to Charlier polynomials
    - As β → ∞ with proper scaling: Hermite polynomials

    See Also
    --------
    Charlier : Limiting case with Poisson weight
    Krawtchouk : Finite support analog
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        beta_init: float = 1.0,
        beta_trainable: bool = True,
        c_init: float = 0.5,
        c_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        degree : int
            Maximum polynomial degree.
        units : int
            Output dimensionality.
        beta_init : float
            Initial value for β (must be > 0).
        beta_trainable : bool
            Whether β is trainable.
        c_init : float
            Initial value for c (must be in (0, 1)).
        c_trainable : bool
            Whether c is trainable.
        input_clip : tuple[float, float] | None
            Optional input clamp.
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        if not beta_init > 0:
            raise ValueError(f"beta_init must be > 0, got {beta_init}")
        if not 0 < c_init < 1:
            raise ValueError(f"c_init must be in (0, 1), got {c_init}")
        
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)
        
        self.beta_init = beta_init
        self.beta_trainable = beta_trainable
        self.c_init = c_init
        self.c_trainable = c_trainable
        self._beta_logits = None
        self._c_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        # β > 0: use softplus with Python inverse for graph-mode
        beta_logit = _inverse_softplus_lower_bound(self.beta_init, lower_bound=0.0)
        self._beta_logits = self.add_weight(
            shape=(),
            initializer=tfk.initializers.Constant(beta_logit),
            name="beta_logits",
            trainable=self.beta_trainable,
        )
        # c ∈ (0, 1): use sigmoid with Python math.log for graph-mode
        c_logit_init = math.log(self.c_init / (1.0 - self.c_init))
        self._c_logits = self.add_weight(
            shape=(),
            initializer=tfk.initializers.Constant(c_logit_init),
            name="c_logits",
            trainable=self.c_trainable,
        )

    @kan_fn
    def pseudo_vandermonde(self, x: tf.Tensor) -> tf.Tensor:
        """Compute Meixner basis using three-term recurrence."""
        beta = softplus_lower_bound(tf.cast(self._beta_logits, x.dtype), lower_bound=0.0)
        c = tf.sigmoid(tf.cast(self._c_logits, x.dtype))
        
        # M_0(x) = 1
        basis = [tf.ones_like(x)]
        
        if self.degree > 0:
            # M_1 from recurrence with n=0
            # A_0 = c*β / (1-c)
            # C_0 = 0
            # -x M_0 = A_0 M_1 - A_0 M_0
            # M_1 = (A_0 - x) / A_0 = 1 - x(1-c)/(c*β)
            one_minus_c = 1.0 - c + 1e-8
            A_0 = c * beta / one_minus_c
            M1 = 1.0 - x / (A_0 + 1e-8)
            basis.append(M1)
        
        for n in range(1, self.degree):
            n_f = tf.cast(n, x.dtype)
            one_minus_c = 1.0 - c + 1e-8
            
            A_n = c * (n_f + beta) / one_minus_c
            C_n = n_f / one_minus_c
            
            # M_{n+1} = [(A_n + C_n - x) M_n - C_n M_{n-1}] / A_n
            M_next = ((A_n + C_n - x) * basis[n] - C_n * basis[n - 1]) / (A_n + 1e-8)
            basis.append(M_next)
        
        return tf.stack(basis, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "beta_init": self.beta_init,
            "beta_trainable": self.beta_trainable,
            "c_init": self.c_init,
            "c_trainable": self.c_trainable,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="Racah")
class Racah(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Racah polynomials.

    The Racah polynomials :math:`R_n(\lambda(x); \alpha, \beta, \gamma, \delta)`
    are the most general family of discrete orthogonal polynomials in the Askey
    scheme. They are orthogonal on a finite set with quadratic lattice.

    The argument transformation is:

    .. math::

        \lambda(x) = x(x + \gamma + \delta + 1)

    Three-term recurrence (DLMF 18.22.5):

    .. math::

        -\lambda(x) R_n(x) = A_n R_{n+1}(x) - (A_n + C_n) R_n(x) + C_n R_{n-1}(x)

    where (with N being the truncation parameter):

    .. math::

        A_n &= \frac{(n+\alpha+1)(n+\beta+\delta+1)(n+\gamma+1)(N-n)}
               {(2n+\alpha+\beta+2)(2n+\alpha+\beta+1)} \\
        C_n &= \frac{n(n+\alpha+\beta-N)(n+\alpha-\gamma)(n+\beta)}
               {(2n+\alpha+\beta+1)(2n+\alpha+\beta)}

    Parameters
    ----------
    alpha_init, beta_init, gamma_init, delta_init : float
        Initial values for the four shape parameters.
    N : int
        Truncation parameter determining support size.

    Notes
    -----
    Racah polynomials appear in:
    - Quantum mechanics (6j-symbols, angular momentum coupling)
    - Representation theory of SU(2)
    - Coding theory and combinatorics

    Special cases:
    - δ = β + N → Dual Hahn polynomials
    - γ = -N - 1 → Hahn polynomials

    References
    ----------
    .. [1] Koekoek, R., Lesky, P. A., & Swarttouw, R. F. (2010).
           Hypergeometric orthogonal polynomials. Springer.

    See Also
    --------
    Hahn : Simpler special case
    Wilson : Continuous analog of Racah
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        alpha_init: float = 0.5,
        alpha_trainable: bool = True,
        beta_init: float = 0.5,
        beta_trainable: bool = True,
        gamma_init: float = 0.5,
        gamma_trainable: bool = True,
        delta_init: float = 0.5,
        delta_trainable: bool = True,
        N: int = 10,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        degree : int
            Maximum polynomial degree. Must be ≤ N.
        units : int
            Output dimensionality.
        alpha_init, beta_init, gamma_init, delta_init : float
            Initial shape parameters (must be > -1 for orthogonality).
        *_trainable : bool
            Whether each parameter is trainable.
        N : int
            Truncation parameter.
        input_clip : tuple[float, float] | None
            Optional input clamp.
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        if N < degree:
            raise ValueError(f"N must be >= degree, got N={N}, degree={degree}")
        
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)
        
        self.alpha_init = alpha_init
        self.alpha_trainable = alpha_trainable
        self.beta_init = beta_init
        self.beta_trainable = beta_trainable
        self.gamma_init = gamma_init
        self.gamma_trainable = gamma_trainable
        self.delta_init = delta_init
        self.delta_trainable = delta_trainable
        self.N = N
        
        self._alpha_logits = None
        self._beta_logits = None
        self._gamma_logits = None
        self._delta_logits = None

    def build(self, input_shape):
        super().build(input_shape)
        # All parameters > -1 for orthogonality
        # Use Python inverse_softplus for graph-mode compatibility
        alpha_logit = _inverse_softplus_lower_bound(self.alpha_init, lower_bound=-1.0)
        self._alpha_logits = self.add_weight(
            shape=(),
            initializer=tfk.initializers.Constant(alpha_logit),
            name="alpha_logits",
            trainable=self.alpha_trainable,
        )
        beta_logit = _inverse_softplus_lower_bound(self.beta_init, lower_bound=-1.0)
        self._beta_logits = self.add_weight(
            shape=(),
            initializer=tfk.initializers.Constant(beta_logit),
            name="beta_logits",
            trainable=self.beta_trainable,
        )
        gamma_logit = _inverse_softplus_lower_bound(self.gamma_init, lower_bound=-1.0)
        self._gamma_logits = self.add_weight(
            shape=(),
            initializer=tfk.initializers.Constant(gamma_logit),
            name="gamma_logits",
            trainable=self.gamma_trainable,
        )
        delta_logit = _inverse_softplus_lower_bound(self.delta_init, lower_bound=-1.0)
        self._delta_logits = self.add_weight(
            shape=(),
            initializer=tfk.initializers.Constant(delta_logit),
            name="delta_logits",
            trainable=self.delta_trainable,
        )

    @kan_fn
    def pseudo_vandermonde(self, x: tf.Tensor) -> tf.Tensor:
        """Compute Racah basis using three-term recurrence."""
        alpha = softplus_lower_bound(tf.cast(self._alpha_logits, x.dtype), lower_bound=-1.0)
        beta = softplus_lower_bound(tf.cast(self._beta_logits, x.dtype), lower_bound=-1.0)
        gamma = softplus_lower_bound(tf.cast(self._gamma_logits, x.dtype), lower_bound=-1.0)
        delta = softplus_lower_bound(tf.cast(self._delta_logits, x.dtype), lower_bound=-1.0)
        N = tf.cast(self.N, x.dtype)
        
        # Transform to λ(x) = x(x + γ + δ + 1)
        lam = x * (x + gamma + delta + 1.0)
        
        # R_0 = 1
        basis = [tf.ones_like(x)]
        
        if self.degree > 0:
            # R_1 from recurrence with n=0
            s = alpha + beta
            A_0_num = (alpha + 1.0) * (beta + delta + 1.0) * (gamma + 1.0) * N
            A_0_den = (s + 2.0) * (s + 1.0) + 1e-8
            A_0 = A_0_num / A_0_den
            # C_0 = 0
            # -λ R_0 = A_0 R_1 - A_0 R_0
            # R_1 = (A_0 - λ) / A_0
            R1 = (A_0 - lam) / (A_0 + 1e-8)
            basis.append(R1)
        
        for n in range(1, self.degree):
            n_f = tf.cast(n, x.dtype)
            s = alpha + beta
            
            # A_n
            A_n_num = (n_f + alpha + 1.0) * (n_f + beta + delta + 1.0) * (n_f + gamma + 1.0) * (N - n_f)
            A_n_den = (2.0 * n_f + s + 2.0) * (2.0 * n_f + s + 1.0) + 1e-8
            A_n = A_n_num / A_n_den
            
            # C_n
            C_n_num = n_f * (n_f + s - N) * (n_f + alpha - gamma) * (n_f + beta)
            C_n_den = (2.0 * n_f + s + 1.0) * (2.0 * n_f + s) + 1e-8
            C_n = C_n_num / C_n_den
            
            # R_{n+1} = [(A_n + C_n - λ) R_n - C_n R_{n-1}] / A_n
            R_next = ((A_n + C_n - lam) * basis[n] - C_n * basis[n - 1]) / (A_n + 1e-8)
            basis.append(R_next)
        
        return tf.stack(basis, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha_init": self.alpha_init,
            "alpha_trainable": self.alpha_trainable,
            "beta_init": self.beta_init,
            "beta_trainable": self.beta_trainable,
            "gamma_init": self.gamma_init,
            "gamma_trainable": self.gamma_trainable,
            "delta_init": self.delta_init,
            "delta_trainable": self.delta_trainable,
            "N": self.N,
        })
        return config
