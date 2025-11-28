## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
import tensorflow as tf

from arnold.utils.compilation import kan_function
from .poly_base import PolynomialBase


tfk = tf.keras
tfkl = tfk.layers
kan_fn = kan_function()  # @kan_fn = @tf.function(jit_compile=True)


@tfk.utils.register_keras_serializable(package="arnold", name="Lucas")
class Lucas(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Lucas polynomials.

    Domain is :math:`\mathbb{R}`; growth is exponential in degree, so clipping
    inputs can help when using large orders.

    The Lucas polynomials are the w-polynomials obtained by setting p(x)=x and q(x)=1 in the Lucas polynomial sequence:

    * :math:`w_{0}(x) = L_{0}(x) = 2`
    * :math:`w_{1}(x) = L_{0}(x) = x`
    * :math:`w_{n+1}(x) = p(x) * w_{n}(x) + q(x) * w_{n-1}(x)` when n >= 1


    It is given explicitly by

    :math:`L_{n}(x) = 2^{-n} * ( (x - \sqrt(x^{2} + 4) )^{n} + (x + \sqrt(x^{2} + 4) )^{n} )`

    See also: https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        """
        Parameters
        ----------
        degree : int
            Maximum polynomial degree.
        units : int
            Output dimensionality.
        input_clip : tuple[float, float] | None
            Optional clamp of inputs before basis evaluation.
        **kwargs :
            Forwarded to :class:`PolynomialBase`.
        """
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Lucas polynomial evaluation.
        
        Recurrence: :math:`L_{n+1}(x) = x L_n(x) + L_{n-1}(x)`.
        """
        L0 = 2.0 * tf.ones_like(x)
        if self.degree == 0:
            return tf.expand_dims(L0, axis=-1)
        
        L1 = x
        if self.degree == 1:
            return tf.stack([L0, L1], axis=-1)
        
        def step(carry, _):
            Ln_1, Ln_2 = carry
            Ln = x * Ln_1 + Ln_2
            return (Ln, Ln_1)  # Only return new carry
        
        carries = tf.scan(
            step,
            tf.range(2, self.degree + 1),
            initializer=(L1, L0)
        )
        # carries[0] has shape (num_steps, batch, input_dim), transpose to (batch, input_dim, num_steps)
        Ls = tf.transpose(carries[0], perm=[1, 2, 0])
        lucas_basis = tf.concat([tf.expand_dims(L0, -1), tf.expand_dims(L1, -1), Ls], axis=-1)
        return lucas_basis


@tfk.utils.register_keras_serializable(package="arnold", name="FermatLucas")
class FermatLucas(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Fermat polynomials.

    Domain is :math:`\mathbb{R}`; behaves similarly to Lucas sequences with
    exponential growth in degree. Use ``input_clip`` for stability when inputs
    are unbounded.

    The Fermat-Lucas polynomials are the w-polynomials obtained by setting p(x)=3*x and q(x)=-2 in the Lucas polynomial sequence:

    * :math:`w_{0}(x) = f_{0}(x) = 2`
    * :math:`w_{1}(x) = f_{1}(x) = 3*x`
    * :math:`w_{n+1}(x) = p(x) * w_{n}(x) + q(x) * w_{n-1}(x)` when n >= 1

    See also: https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Fermat-Lucas polynomial evaluation.
        
        Recurrence: :math:`f_{n+1}(x) = 3x f_n(x) - 2 f_{n-1}(x)`.
        """
        f0 = 2.0 * tf.ones_like(x)
        if self.degree == 0:
            return tf.expand_dims(f0, axis=-1)
        
        f1 = 3.0 * x
        if self.degree == 1:
            return tf.stack([f0, f1], axis=-1)
        
        def step(carry, _):
            fn_1, fn_2 = carry
            fn = 3.0 * x * fn_1 - 2.0 * fn_2
            return (fn, fn_1)  # Only return new carry
        
        carries = tf.scan(
            step,
            tf.range(2, self.degree + 1),
            initializer=(f1, f0)
        )
        # carries[0] has shape (num_steps, batch, input_dim), transpose to (batch, input_dim, num_steps)
        fs = tf.transpose(carries[0], perm=[1, 2, 0])
        fermat_lucas_basis = tf.concat([tf.expand_dims(f0, -1), tf.expand_dims(f1, -1), fs], axis=-1)
        return fermat_lucas_basis


@tfk.utils.register_keras_serializable(package="arnold", name="Fermat")
class Fermat(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Fermat polynomials.

    Domain is :math:`\mathbb{R}`; growth is exponential in degree, so clipping
    inputs can help when using large orders.

    The Fermat polynomials are the w-polynomials obtained by setting p(x)=3*x and q(x)=-2 in the Lucas polynomial sequence:

    * :math:`w_{0}(x) = F_{0}(x) = 0`
    * :math:`w_{1}(x) = F_{1}(x) = 1`
    * :math:`w_{n+1}(x) = p(x) * w_{n}(x) + q(x) * w_{n-1}(x)` when n >= 1

    See also: https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Fermat polynomial evaluation.
        
        Recurrence: :math:`F_{n+1}(x) = 3x F_n(x) - 2 F_{n-1}(x)`.
        """
        F0 = tf.zeros_like(x)
        if self.degree == 0:
            return tf.expand_dims(F0, axis=-1)
        
        F1 = tf.ones_like(x)
        if self.degree == 1:
            return tf.stack([F0, F1], axis=-1)
        
        def step(carry, _):
            Fn_1, Fn_2 = carry
            Fn = 3.0 * x * Fn_1 - 2.0 * Fn_2
            return (Fn, Fn_1)  # Only return new carry
        
        carries = tf.scan(
            step,
            tf.range(2, self.degree + 1),
            initializer=(F1, F0)
        )
        # carries[0] has shape (num_steps, batch, input_dim), transpose to (batch, input_dim, num_steps)
        Fs = tf.transpose(carries[0], perm=[1, 2, 0])
        fermat_basis = tf.concat([tf.expand_dims(F0, -1), tf.expand_dims(F1, -1), Fs], axis=-1)
        return fermat_basis


@tfk.utils.register_keras_serializable(package="arnold", name="JacobsthalLucas")
class JacobsthalLucas(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Jacobsthal-Lucas polynomials.

    Domain is :math:`\mathbb{R}`; growth is exponential in degree, so clipping
    inputs can help when using large orders.

    The Jacobsthal-lucas polynomials are the w-polynomials obtained by setting p(x)=1 and q(x)=2*x in the Lucas polynomial sequence:

    * :math:`w_{0}(x) = j_{0}(x) = 2`
    * :math:`w_{1}(x) = j_{1}(x) = 1`
    * :math:`w_{n+1}(x) = p(x) * w_{n}(x) + q(x) * w_{n-1}(x)` when n >= 1

    See also: https://www.fq.math.ca/Scanned/35-2/horadam.pdf
    See also: https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Jacobsthal-Lucas polynomial evaluation.
        
        Recurrence: :math:`j_{n+1}(x) = j_n(x) + 2x j_{n-1}(x)`.
        """
        j0 = 2.0 * tf.ones_like(x)
        if self.degree == 0:
            return tf.expand_dims(j0, axis=-1)
        
        j1 = tf.ones_like(x)
        if self.degree == 1:
            return tf.stack([j0, j1], axis=-1)
        
        def step(carry, _):
            jn_1, jn_2 = carry
            jn = jn_1 + 2.0 * x * jn_2
            return (jn, jn_1)  # Only return new carry
        
        carries = tf.scan(
            step,
            tf.range(2, self.degree + 1),
            initializer=(j1, j0)
        )
        # carries[0] has shape (num_steps, batch, input_dim), transpose to (batch, input_dim, num_steps)
        js = tf.transpose(carries[0], perm=[1, 2, 0])
        jacobsthal_lucas_basis = tf.concat([tf.expand_dims(j0, -1), tf.expand_dims(j1, -1), js], axis=-1)
        return jacobsthal_lucas_basis


@tfk.utils.register_keras_serializable(package="arnold", name="Jacobsthal")
class Jacobsthal(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Jacobsthal polynomials.

    Domain is :math:`\mathbb{R}`; growth is exponential in degree, so clipping
    inputs can help when using large orders.

    The Jacobsthal polynomials are the w-polynomials obtained by setting p(x)=1 and q(x)=2*x in the Lucas polynomial sequence:

    * :math:`w_{0}(x) = J_{0}(x) = 0`
    * :math:`w_{1}(x) = J_{1}(x) = 1`
    * :math:`w_{n+1}(x) = p(x) * w_{n}(x) + q(x) * w_{n-1}(x)` when n >= 1

    See also: https://www.fq.math.ca/Scanned/35-2/horadam.pdf
    See also: https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Jacobsthal polynomial evaluation.
        
        Recurrence: :math:`J_{n+1}(x) = J_n(x) + 2x J_{n-1}(x)`.
        """
        J0 = tf.zeros_like(x)
        if self.degree == 0:
            return tf.expand_dims(J0, axis=-1)
        
        J1 = tf.ones_like(x)
        if self.degree == 1:
            return tf.stack([J0, J1], axis=-1)
        
        def step(carry, _):
            Jn_1, Jn_2 = carry
            Jn = Jn_1 + 2.0 * x * Jn_2
            return (Jn, Jn_1)  # Only return new carry
        
        carries = tf.scan(
            step,
            tf.range(2, self.degree + 1),
            initializer=(J1, J0)
        )
        # carries[0] has shape (num_steps, batch, input_dim), transpose to (batch, input_dim, num_steps)
        Js = tf.transpose(carries[0], perm=[1, 2, 0])
        jacobsthal_basis = tf.concat([tf.expand_dims(J0, -1), tf.expand_dims(J1, -1), Js], axis=-1)
        return jacobsthal_basis


@tfk.utils.register_keras_serializable(package="arnold", name="PellLucas")
class PellLucas(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Pell-Lucas polynomials.

    Domain is :math:`\mathbb{R}`; growth is exponential in degree, so clipping
    inputs can help when using large orders.

    The Pell-Lucas polynomials are the w-polynomials obtained by setting p(x)=2*x and q(x)=1 in the Lucas polynomial sequence:

    * :math:`w_{0}(x) = Q_{0}(x) = 2`
    * :math:`w_{1}(x) = Q_{0}(x) = 2x`
    * :math:`w_{n+1}(x) = p(x) * w_{n}(x) + q(x) * w_{n-1}(x)` when n >= 1

    See also: https://www.mathstat.dal.ca/FQ/Scanned/23-1/horadam.pdf
    See also: https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Pell-Lucas polynomial evaluation.
        
        Recurrence: :math:`Q_{n+1}(x) = 2x Q_n(x) + Q_{n-1}(x)`.
        """
        Q0 = 2.0 * tf.ones_like(x)
        if self.degree == 0:
            return tf.expand_dims(Q0, axis=-1)
        
        Q1 = 2.0 * x
        if self.degree == 1:
            return tf.stack([Q0, Q1], axis=-1)
        
        def step(carry, _):
            Qn_1, Qn_2 = carry
            Qn = 2.0 * x * Qn_1 + Qn_2
            return (Qn, Qn_1)  # Only return new carry
        
        carries = tf.scan(
            step,
            tf.range(2, self.degree + 1),
            initializer=(Q1, Q0)
        )
        # carries[0] has shape (num_steps, batch, input_dim), transpose to (batch, input_dim, num_steps)
        Qs = tf.transpose(carries[0], perm=[1, 2, 0])
        pell_lucas_basis = tf.concat([tf.expand_dims(Q0, -1), tf.expand_dims(Q1, -1), Qs], axis=-1)
        return pell_lucas_basis


@tfk.utils.register_keras_serializable(package="arnold", name="Pell")
class Pell(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Pell polynomials.

    The Pell-Lucas polynomials are the w-polynomials obtained by setting p(x)=2*x and q(x)=1 in the Lucas polynomial sequence:

    * :math:`w_{0}(x) = P_{0}(x) = 0`
    * :math:`w_{1}(x) = P_{0}(x) = 1`
    * :math:`w_{n+1}(x) = p(x) * w_{n}(x) + q(x) * w_{n-1}(x)` when n >= 1

    See also: https://www.mathstat.dal.ca/FQ/Scanned/23-1/horadam.pdf
    See also: https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Pell polynomial evaluation.
        
        Recurrence: :math:`P_{n+1}(x) = 2x P_n(x) + P_{n-1}(x)`.
        """
        P0 = tf.zeros_like(x)
        if self.degree == 0:
            return tf.expand_dims(P0, axis=-1)
        
        P1 = tf.ones_like(x)
        if self.degree == 1:
            return tf.stack([P0, P1], axis=-1)
        
        def step(carry, _):
            Pn_1, Pn_2 = carry
            Pn = 2.0 * x * Pn_1 + Pn_2
            return (Pn, Pn_1)  # Only return new carry
        
        carries = tf.scan(
            step,
            tf.range(2, self.degree + 1),
            initializer=(P1, P0)
        )
        # carries[0] has shape (num_steps, batch, input_dim), transpose to (batch, input_dim, num_steps)
        Ps = tf.transpose(carries[0], perm=[1, 2, 0])
        pell_basis = tf.concat([tf.expand_dims(P0, -1), tf.expand_dims(P1, -1), Ps], axis=-1)
        return pell_basis
