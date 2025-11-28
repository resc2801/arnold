## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
import tensorflow as tf

from arnold.utils.compilation import kan_function
from .poly_base import PolynomialBase


tfk = tf.keras
tfkl = tfk.layers
kan_fn = kan_function()  # @kan_fn = @tf.function(jit_compile=True)


@tfk.utils.register_keras_serializable(package="arnold", name="Fibonacci")
class Fibonacci(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Fibonacci polynomials.

    The Fibonacci polynomials are the w-polynomials obtained by setting p(x)=x and q(x)=1 in the Lucas polynomial sequence:

    * :math:`w_{0}(x) = F_{0}(x) = 0`
    * :math:`w_{1}(x) = F_{1}(x) = 1`
    * :math:`w_{n+1}(x) = x * w_{n}(x) + w_{n-1}(x)` when n >= 1

    See also: https://en.wikipedia.org/wiki/Fibonacci_polynomials#Definition
    See also: https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Fibonacci polynomial evaluation.
        
        Recurrence: :math:`F_{n+1}(x) = x F_n(x) + F_{n-1}(x)`.
        """
        F0 = tf.zeros_like(x)
        F1 = tf.ones_like(x)
        
        if self.degree == 0:
            return tf.expand_dims(F0, axis=-1)
        if self.degree == 1:
            return tf.stack([F0, F1], axis=-1)
        
        def step(carry, _):
            Fn_1, Fn_2 = carry
            Fn = x * Fn_1 + Fn_2
            return (Fn, Fn_1)
        
        basis_list = [F0, F1]
        Fn_1, Fn_2 = F1, F0
        for _ in range(2, self.degree + 1):
            Fn_1, Fn_2 = step((Fn_1, Fn_2), None)
            basis_list.append(Fn_1)
        
        return tf.stack(basis_list, axis=-1)


@tfk.utils.register_keras_serializable(package="arnold", name="Heptanacci")
class Heptanacci(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Heptanacci polynomials.

    Heptanacci polynomials are a generalization of the Fibonacci polynomials.
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Heptanacci polynomial evaluation.
        
        7-term recurrence: :math:`H_n = x H_{n-1} + \sum_{i=n-6}^{n-2} H_i`.
        """
        H0 = tf.zeros_like(x)
        H1 = tf.ones_like(x)
        H2 = x
        H3 = x
        H4 = x**2
        H5 = x**2
        H6 = x**3
        
        if self.degree <= 6:
            basis_list = [H0, H1, H2, H3, H4, H5, H6][:(self.degree + 1)]
            return tf.stack(basis_list, axis=-1)
        
        def step(carry, _):
            Hn_1, Hn_2, Hn_3, Hn_4, Hn_5, Hn_6, Hn_7 = carry
            Hn = x * Hn_1 + Hn_2 + Hn_3 + Hn_4 + Hn_5 + Hn_6 + Hn_7
            return (Hn, Hn_1, Hn_2, Hn_3, Hn_4, Hn_5, Hn_6)
        
        basis_list = [H0, H1, H2, H3, H4, H5, H6]
        carry = (H6, H5, H4, H3, H2, H1, H0)
        for _ in range(7, self.degree + 1):
            carry = step(carry, None)
            basis_list.append(carry[0])
        
        return tf.stack(basis_list, axis=-1)


@tfk.utils.register_keras_serializable(package="arnold", name="Hexanacci")
class Hexanacci(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Heptanacci polynomials.

    Hexanacci polynomials are a generalization of the Fibonacci polynomials.
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Hexanacci polynomial evaluation.
        
        6-term recurrence: :math:`H_n = x H_{n-1} + \sum_{i=n-5}^{n-2} H_i`.
        """
        H0 = tf.zeros_like(x)
        H1 = tf.ones_like(x)
        H2 = x
        H3 = x**2
        H4 = x**2
        H5 = x**3
        
        if self.degree <= 5:
            basis_list = [H0, H1, H2, H3, H4, H5][:(self.degree + 1)]
            return tf.stack(basis_list, axis=-1)
        
        def step(carry, _):
            Hn_1, Hn_2, Hn_3, Hn_4, Hn_5, Hn_6 = carry
            Hn = x * Hn_1 + Hn_2 + Hn_3 + Hn_4 + Hn_5 + Hn_6
            return (Hn, Hn_1, Hn_2, Hn_3, Hn_4, Hn_5)
        
        basis_list = [H0, H1, H2, H3, H4, H5]
        carry = (H5, H4, H3, H2, H1, H0)
        for _ in range(6, self.degree + 1):
            carry = step(carry, None)
            basis_list.append(carry[0])
        
        return tf.stack(basis_list, axis=-1)


@tfk.utils.register_keras_serializable(package="arnold", name="Octanacci")
class Octanacci(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Heptanacci polynomials.

    Octanacci polynomials are a generalization of the Fibonacci polynomials.
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Octanacci polynomial evaluation.
        
        8-term recurrence: :math:`O_n = x O_{n-1} + \sum_{i=n-7}^{n-2} O_i`.
        """
        O0 = tf.zeros_like(x)
        O1 = tf.ones_like(x)
        O2 = x
        O3 = x**2
        O4 = x**2
        O5 = x**3
        O6 = x**3
        O7 = x**4
        
        if self.degree <= 7:
            basis_list = [O0, O1, O2, O3, O4, O5, O6, O7][:(self.degree + 1)]
            return tf.stack(basis_list, axis=-1)
        
        def step(carry, _):
            On_1, On_2, On_3, On_4, On_5, On_6, On_7, On_8 = carry
            On = x * On_1 + On_2 + On_3 + On_4 + On_5 + On_6 + On_7 + On_8
            return (On, On_1, On_2, On_3, On_4, On_5, On_6, On_7)
        
        basis_list = [O0, O1, O2, O3, O4, O5, O6, O7]
        carry = (O7, O6, O5, O4, O3, O2, O1, O0)
        for _ in range(8, self.degree + 1):
            carry = step(carry, None)
            basis_list.append(carry[0])
        
        return tf.stack(basis_list, axis=-1)


@tfk.utils.register_keras_serializable(package="arnold", name="Pentanacci")
class Pentanacci(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Heptanacci polynomials.

    Pentanacci polynomials are a generalization of the Fibonacci polynomials.
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Pentanacci polynomial evaluation.
        
        5-term recurrence: :math:`P_n = x P_{n-1} + \sum_{i=n-4}^{n-2} P_i`.
        """
        P0 = tf.zeros_like(x)
        P1 = tf.ones_like(x)
        P2 = x
        P3 = x
        P4 = x**2
        
        if self.degree <= 4:
            basis_list = [P0, P1, P2, P3, P4][:(self.degree + 1)]
            return tf.stack(basis_list, axis=-1)
        
        def step(carry, _):
            Pn_1, Pn_2, Pn_3, Pn_4, Pn_5 = carry
            Pn = x * Pn_1 + Pn_2 + Pn_3 + Pn_4 + Pn_5
            return (Pn, Pn_1, Pn_2, Pn_3, Pn_4)
        
        basis_list = [P0, P1, P2, P3, P4]
        carry = (P4, P3, P2, P1, P0)
        for _ in range(5, self.degree + 1):
            carry = step(carry, None)
            basis_list.append(carry[0])
        
        return tf.stack(basis_list, axis=-1)


@tfk.utils.register_keras_serializable(package="arnold", name="Tetranacci")
class Tetranacci(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Heptanacci polynomials.

    Tetranacci polynomials are a generalization of the Fibonacci polynomials.
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Tetranacci polynomial evaluation.
        
        4-term recurrence: :math:`T_n = x T_{n-1} + T_{n-2} + T_{n-3} + T_{n-4}`.
        """
        T0 = tf.zeros_like(x)
        T1 = tf.ones_like(x)
        T2 = x
        T3 = x**2
        
        if self.degree <= 3:
            basis_list = [T0, T1, T2, T3][:(self.degree + 1)]
            return tf.stack(basis_list, axis=-1)
        
        def step(carry, _):
            Tn_1, Tn_2, Tn_3, Tn_4 = carry
            Tn = x * Tn_1 + Tn_2 + Tn_3 + Tn_4
            return (Tn, Tn_1, Tn_2, Tn_3)
        
        basis_list = [T0, T1, T2, T3]
        carry = (T3, T2, T1, T0)
        for _ in range(4, self.degree + 1):
            carry = step(carry, None)
            basis_list.append(carry[0])
        
        return tf.stack(basis_list, axis=-1)


@tfk.utils.register_keras_serializable(package="arnold", name="Tribonacci")
class Tribonacci(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Tribonacci polynomials.

    The Tribonacci polynomials :math:`T_n(x)` are a generalization of the 
    Fibonacci polynomials using a 3-term sum recurrence:

    .. math::

        T_0(x) = 0, \quad T_1(x) = 1, \quad T_2(x) = x

    .. math::

        T_n(x) = x \cdot T_{n-1}(x) + T_{n-2}(x) + T_{n-3}(x), \quad n \geq 3

    The sequence of Tribonacci polynomials begins:
    
    - :math:`T_0 = 0`
    - :math:`T_1 = 1`
    - :math:`T_2 = x`
    - :math:`T_3 = x^2 + 1`
    - :math:`T_4 = x^3 + 2x`
    - :math:`T_5 = x^4 + 3x^2 + 1`

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.
    units : int
        Output dimension.

    Notes
    -----
    - Tribonacci numbers are :math:`T_n(1) = 0, 1, 1, 2, 4, 7, 13, 24, \ldots`
    - The characteristic equation is :math:`t^3 = t^2 + t + 1`
    - Related to ternary representations and combinatorics

    References
    ----------
    .. [1] OEIS A000073 - Tribonacci numbers
    .. [2] Spickerman, W.R. (1982). "Binet's formula for the Tribonacci sequence"
    """

    def __init__(self, degree: int, *, units: int, input_clip=None, **kwargs):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)

    @kan_fn
    def pseudo_vandermonde(self, x):
        r"""
        Vectorized Tribonacci polynomial evaluation.
        
        3-term recurrence: :math:`T_n = x T_{n-1} + T_{n-2} + T_{n-3}`.
        
        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        tf.Tensor
            Basis tensor of shape (batch, input_dim, degree+1).
        """
        T0 = tf.zeros_like(x)
        T1 = tf.ones_like(x)
        T2 = x
        
        if self.degree <= 2:
            basis_list = [T0, T1, T2][:(self.degree + 1)]
            return tf.stack(basis_list, axis=-1)
        
        def step(carry, _):
            Tn_1, Tn_2, Tn_3 = carry
            Tn = x * Tn_1 + Tn_2 + Tn_3
            return (Tn, Tn_1, Tn_2)
        
        basis_list = [T0, T1, T2]
        carry = (T2, T1, T0)
        for _ in range(3, self.degree + 1):
            carry = step(carry, None)
            basis_list.append(carry[0])
        
        return tf.stack(basis_list, axis=-1)
