from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

from .poly_base import PolynomialBase
from arnold.math import generalized_hypergeometric

tfk = tf.keras
tfkl = tfk.layers


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
            *args,
            a_init: float | None = None, a_trainable=True, 
            q_init: float | None = None, q_trainable=True,
            **kwargs):
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

        :param a_init: Initial value for the a parameter of the AlSalamCarlitz polynomials. Defaults to None (a initialized to 0).
        :type a_init: float | None = None

        :param a_trainable: Flag indicating whether alpha is a trainable parameter. Defaults to True
        :type a_trainable: bool

        :param q_init: Initial value for the q parameter of the AlSalamCarlitz polynomials. Defaults to None (q initialized to 1).
        :type q_init: float | None = None

        :param q_trainable: Flag indicating whether q is a trainable parameter. Defaults to True
        :type q_trainable: bool
        """
        super().__init__(*args, **kwargs)

        self.a_init = a_init
        self.a_trainable = a_trainable
        self.q_init = q_init
        self.q_trainable = q_trainable
        
        self.a = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.a_init) if self.a_init else tfk.initializers.Zeros(),
            name='a',
            trainable=self.a_trainable
        )

        self.q = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.q_init) if self.q_init else tfk.initializers.Ones(),
            name='q',
            trainable=self.q_trainable
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "a_init": self.a_init,
            "a_trainable": self.a_trainable,
            "q_init": self.q_init,
            "q_trainable": self.q_trainable,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="AlSalamCarlitz1st")
class AlSalamCarlitz1st(AlSalamCarlitz):
    r"""
    Kolmogorov-Arnold Network layer using the Al-Salam-Carlitz polynomials :math:`U^{(a)}_{n} (x;q)`.

    These polynomials satisfy the three-term recurrence relation 

    * :math:`U^{(a)}_{-1} (x;q) = 0`
    * :math:`U^{(a)}_{0} (x;q) = 1`
    * :math:`U^{(a)}_{n+1} (x;q) = (x - (1 + a) q^{n}) U^{(a)}_{n} (x;q) + a q^{n-1} (1 - q^{n}) U^{(a)}_{n-1} (x;q)`

    See also: https://core.ac.uk/download/pdf/82826366.pdf
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        al_salam_carlitz_basis = [
            # $U^{(a)}_{0} (x;q) = 1$
            tf.ones_like(x)
        ]

        if self.degree > 0:
            # $U^{(a)}_{1} (x;q) = x - 1$
            al_salam_carlitz_basis.append(x - 1.0)

        for n in range(2, self.degree + 1):
            # $U^{(a)}_{n+1} (x;q) = (x - (1 + a) * q^{n}) * U^{(a)}_{n} (x;q) + a*q^{n-1} * (1 - q^{n}) * U^{(a)}_{n-1} (x;q)$
            al_salam_carlitz_basis.append(
                (x - (1 + self.a) * tf.pow(self.q, n-1)) * al_salam_carlitz_basis[n-1] + self.a * tf.pow(self.q, n-2) * (1 - tf.pow(self.q, n-2)) * al_salam_carlitz_basis[n-2]
            )

        return tf.stack(al_salam_carlitz_basis, axis=-1) 


@tfk.utils.register_keras_serializable(package="arnold", name="AlSalamCarlitz2nd")
class AlSalamCarlitz2nd(AlSalamCarlitz):
    r"""
    Kolmogorov-Arnold Network layer using the Al-Salam-Carlitz polynomials :math:`V^{(a)}_{n} (x;q)`.

    There is a straightforward relationship between :math:`U^{(a)}_{n} (x;q)` and :math:`V^{(a)}_{n} (x;q)`:

    :math:`U^{(a)}_{n} (x; 1/q) = V^{(a)}_{n} (x;q)`

    See: Chihara, T.S. An Introduction to Orthogonal Polynomials; Mathematics Applied Series 13; Gordon and Breach: New York, NY, USA, 1978. Chapter VI, §10, pp. 195–198
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):

        al_salam_carlitz_basis = [
            tf.ones_like(x)
        ]

        if self.degree > 0:
            al_salam_carlitz_basis.append(x - 1.0)

        for n in range(2, self.degree + 1):
            al_salam_carlitz_basis.append(
                (x - (1 + self.a) * tf.pow(1.0 / self.q, n-1)) * al_salam_carlitz_basis[n-1] + self.a * tf.pow(1.0 / self.q, n-2) * (1 - tf.pow(1.0 / self.q, n-2)) * al_salam_carlitz_basis[n-2]
            )

        return tf.stack(al_salam_carlitz_basis, axis=-1)


@tfk.utils.register_keras_serializable(package="arnold", name="AskeyWilson")
class AskeyWilson(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using the Askey-Wilson polynomials.
    """

    def __init__(
            self, 
            *args,
            a_init: float | None = None, a_trainable=True, 
            b_init: float | None = None, b_trainable=True,
            c_init: float | None = None, c_trainable=True, 
            d_init: float | None = None, d_trainable=True, 
            q_init: float | None = None, q_trainable=True,
            **kwargs):
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

        :param a_init: Initial value for the a parameter of the AskeyWilson polynomials. Defaults to None (a initialized to RandomNormal).
        :type a_init: float | None = None

        :param a_trainable: Flag indicating whether a is a trainable parameter. Defaults to True
        :type a_trainable: bool

        :param b_init: Initial value for the b parameter of the AskeyWilson polynomials. Defaults to None (a initialized to RandomNormal).
        :type b_init: float | None = None

        :param b_trainable: Flag indicating whether b is a trainable parameter. Defaults to True
        :type b_trainable: bool

        :param c_init: Initial value for the c parameter of the AskeyWilson polynomials. Defaults to None (a initialized to RandomNormal).
        :type c_init: float | None = None

        :param c_trainable: Flag indicating whether c is a trainable parameter. Defaults to True
        :type c_trainable: bool

        :param d_init: Initial value for the parameter d of the AskeyWilson polynomials. Defaults to None (a initialized to RandomNormal).
        :type d_init: float | None = None

        :param d_trainable: Flag indicating whether d is a trainable parameter. Defaults to True
        :type d_trainable: bool

        :param q_init: Initial value for the q parameter of the AskeyWilson polynomials. Defaults to None (q initialized to RandomNormal).
        :type q_init: float | None = None
        
        :param q_trainable: Flag indicating whether q is a trainable parameter. Defaults to True
        :type q_trainable: bool
        """ 
        super().__init__(*args, **kwargs)

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

        self.a = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.a_init) if self.a_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='a',
            trainable=self.a_trainable
        )

        self.b = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.b_init) if self.b_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='b',
            trainable=self.b_trainable
        )

        self.c = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.c_init) if self.c_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='c',
            trainable=self.c_trainable
        )

        self.d = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.d_init) if self.d_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='d',
            trainable=self.d_trainable
        )

        self.q = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.q_init) if self.q_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='q',
            trainable=self.q_trainable
        )

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        askey_wilson_basis = [
            tf.ones_like(x)
        ]

        if self.degree > 0:
            askey_wilson_basis.append((2 * (1 + self.a * self.b * self.q) * x - (self.a + self.b) * (1 + self.c * self.d * self.q)) / (1 + self.a * self.b * self.c * self.d * self.q**2))

        for n in range(2, self.degree + 1):
            An = (1 - self.a * self.b * self.q**(n-1)) * (1 - self.c * self.d * self.q**(n-1)) * (1 - self.a * self.b * self.c * self.d * self.q**(2*n-2))
            An /= (1 - self.a * self.b * self.c * self.d * self.q**(2*n-1)) * (1 - self.a * self.b * self.c * self.d * self.q**(2*n))
            Cn = (1 - self.q**n) * (1 - self.a * self.b * self.q**(n-1)) * (1 - self.c * self.d * self.q**(n-1)) * (1 - self.a * self.b * self.c * self.d * self.q**(2*n-2))
            Cn /= (1 - self.a * self.b * self.c * self.d * self.q**(2*n-2)) * (1 - self.a * self.b * self.c * self.d * self.q**(2*n-1))
            askey_wilson_basis.append(
                ((2 * x - An) * askey_wilson_basis[n-1] - Cn * askey_wilson_basis[n-2]) / (1 - self.q**n)
            )

        return tf.stack(askey_wilson_basis, axis=-1) 
    
    def get_config(self):
        config = super().get_config()
        config.update({
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
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="BannaiIto")
class BannaiIto(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using the Bannai-Ito polynomials.
    """

    def __init__(
            self, 
            *args,
            a_init: float | None = None, a_trainable=True, 
            b_init: float | None = None, b_trainable=True, 
            c_init: float | None = None, c_trainable=True,
            **kwargs):
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
        :param a_init: Initial value for the a parameter of the BannaiIto polynomials. Defaults to None (a initialized to RandomNormal).
        :type a_init: float | None = None
        :param a_trainable: Flag indicating whether a is a trainable parameter. Defaults to True
        :type a_trainable: bool
        :param b_init: Initial value for the b parameter of the BannaiIto polynomials. Defaults to None (a initialized to RandomNormal).
        :type b_init: float | None = None
        :param b_trainable: Flag indicating whether b is a trainable parameter. Defaults to True
        :type b_trainable: bool
        :param c_init: Initial value for the c parameter of the BannaiIto polynomials. Defaults to None (a initialized to RandomNormal).
        :type c_init: float | None = None
        :param c_trainable: Flag indicating whether c is a trainable parameter. Defaults to True
        :type c_trainable: bool
        """ 
        super().__init__(*args, **kwargs)

        self.a_init = a_init
        self.a_trainable = a_trainable
        self.b_init = b_init
        self.b_trainable = b_trainable
        self.c_init = c_init
        self.c_trainable = c_trainable

        self.a = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.a_init) if self.a_init else tfk.initializers.Zeros(),
            name='a',
            trainable=self.a_trainable
        )

        self.b = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.b_init)  if self.b_init else tfk.initializers.Zeros(),
            name='b',
            trainable=self.b_trainable
        )

        self.c = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.c_init) if self.c_init else tfk.initializers.Zeros(),
            name='c',
            trainable=self.c_trainable
        )

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        bannai_ito_basis = [tf.ones_like(x)]

        if self.degree > 0:
            bannai_ito_basis.append(
                (x - self.a) / (self.b + self.c + 1.0)
            )

        for n in range(2, self.degree + 1):
            An = (2 * n + self.b + self.c - 1) * (2 * n + self.b + self.c) / (2 * (n + self.b + self.c))
            Cn = -(n + self.b - 1) * (n + self.c - 1) / (2 * (n + self.b + self.c))

            bannai_ito_basis.append(
                ((x - An) * bannai_ito_basis[n-1] - Cn * bannai_ito_basis[n - 2]) / (n + self.b + self.c)
            )

        return tf.stack(bannai_ito_basis, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "a_init": self.a_init,
            "a_trainable": self.a_trainable,
            "b_init": self.b_init,
            "b_trainable": self.b_trainable,
            "c_init": self.c_init,
            "c_trainable": self.c_trainable,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="Bessel")
class Bessel(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Bessel polynomials.

    The Bessel polynomials are generated by the three-term recurrence relation:

    * :math:`y_{0}(x) = 1`
    * :math:`y_{1}(x) = x + 1`
    * :math:`y_{n}(x) = (2n - 1) * x * y_{n-1}(x) + y_{n-2}(x)` when n >= 2
    
    See also: https://en.wikipedia.org/wiki/Bessel_polynomials#Recursion
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        # :math:`y_{0}(x) = 1`
        bessel_basis = [ tf.ones_like(x) ]

        if self.degree > 0:
            # :math:`y_{1}(x) = x + 1`
            bessel_basis.append(x + 1.0)

        for n in range(2, self.degree + 1):
            # :math:`y_{n}(x) = (2n - 1) * x * y_{n-1}(x) + y_{n-2}(x)` when n >= 2
            bessel_basis.append(
                (2 * n - 1) * x * bessel_basis[n-1] + bessel_basis[n-2]
            )

        return tf.stack(bessel_basis, axis=-1)


@tfk.utils.register_keras_serializable(package="arnold", name="Charlier")
class Charlier(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using the Charlier polynomials.

    The Charlier polynomials are generated by the three-term recurrence relation:

    * :math:`C_{-1; a}(x) = 0`
    * :math:`C_{0; a}(x) = 1`
    * :math:`x * C_{n}(x; a) = C_{n+1}(x; a) + (n + a) * C_{n}(x; a) + a * n * C_{n-1}(x; a)` 

    for :math:`a>0`.

    See: https://arxiv.org/pdf/1901.06041, eq. 1.4
    """

    def __init__(
            self, 
            *args,
            a_init: float | None = None, a_trainable=True, 
            **kwargs):
        
        super().__init__(*args, **kwargs)
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

        :param a_init: Initial value for the a parameter of the Charlier polynomials. Defaults to None (a initialized to RandomNormal).
        :type a_init: float | None = None

        :param a_trainable: Flag indicating whether a is a trainable parameter. Defaults to True
        :type a_trainable: bool
        """ 
        self.a_init = a_init
        self.a_trainable = a_trainable

        self.a = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.a_init) if self.a_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='a',
            trainable=self.a_trainable,
            regularizer=None,
            constraint=None,
        )

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):

        # :math:`a > 0`
        a = tf.exp(self.a)
        
        # :math:`C_{-1; a}(x) = 0`
        # :math:`C_{0; a}(x) = 1`
        charlier_basis = [
            tf.ones_like(x)
        ]

        if self.degree > 0:
            # :math`C_{1}(x; a) = (x - (n + a)) * C_{0}(x; a) - a*n*C_{-1}(x; a)`
            charlier_basis.append(
                (x - (1 + a))  
            )
        
        for n in range(2, self.degree + 1):
            # :math:`x * C_{n}(x; a) = C_{n+1}(x; a) + (n + a) * C_{n}(x; a) + a * n * C_{n-1}(x; a)` 
            charlier_basis.append(
                (x - (n + a)) * charlier_basis[n-1] - a * n * charlier_basis[n-2]
            )

        return tf.stack(charlier_basis, axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "a_init": self.a_init,
            "a_trainable": self.a_trainable,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="Chebyshev")
class Chebyshev(PolynomialBase, ABC):
    r"""
    Abstract base class for Kolmogorov-Arnold Network layer using Chebyshev polynomial basis.

    TODO: check https://www.mathematik.uni-kassel.de/~koepf/Publikationen/cheby.pdf
    """
    def __init__(
            self, 
            *args,
            **kwargs):
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


@tfk.utils.register_keras_serializable(package="arnold", name="Chebyshev1st")
class Chebyshev1st(Chebyshev):
    r"""
    Kolmogorov-Arnold Network layer using 1st kind Chebyshev polynomials in trigonometric formulation 
    
    .. math::
        :nowrap:

        \begin{equation}
        T_{n}(x) = \cos(n \arccos(x)) , \,  \lvert x \rvert \leq 1
        \end{equation}


    See: https://core.ac.uk/download/pdf/82763706.pdf   
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        # Reshape to (batch_size, input_dim, self.degree + 1)
        x = tf.reshape(x, (-1, self.input_dim, 1))
        x = tf.tile(x, (1, 1, self.degree + 1))
        
        # Prepare :math:`T_{n}(x) = \cos(n * \arccos(x))` 
        x = tf.math.acos(x)
        x = tf.multiply(x, self.arange)
        x = tf.math.cos(x)

        return x
        

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
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        # Reshape to (batch_size, input_dim, self.degree + 1)
        x = tf.reshape(x, (-1, self.input_dim, 1))
        x = tf.tile(x, (1, 1, self.degree + 1))

        # Prepare :math:`U_{n}(x) := `\frac{\sin((n+1) * \arccos(x))}{\sin(\arccos(x))}` 
        x = tf.math.acos(x)
        x = tf.math.divide(
            tf.math.sin(tf.multiply(x, self.arange)),
            tf.math.sin(x)
        )

        return x


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
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        # Reshape to (batch_size, input_dim, self.degree + 1)
        x = tf.reshape(x, (-1, self.input_dim, 1))
        x = tf.tile(x, (1, 1, self.degree + 1))

        # Prepare :math:`V_{n}(x) := `\frac{\cos((n + \tfrac{1}{2}) * \arccos(x))}{\cos(\tfrac{1}{2} * \arccos(x))}` 
        x = tf.math.acos(x)
        x = tf.math.divide(
            tf.math.cos(tf.multiply(x, self.arange)),
            tf.math.cos(
                tf.multiply(0.5, x)
            )
        )

        return x
    

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
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        # Reshape to (batch_size, input_dim, self.degree + 1)
        x = tf.reshape(x, (-1, self.input_dim, 1))
        x = tf.tile(x, (1, 1, self.degree + 1))

        # Prepare :math:`W_{n}(x) := `\frac{\sin((n + \tfrac{1}{2}) * \arccos(x))}{\sin(\tfrac{1}{2} * \arccos(x))}` 
        x = tf.math.acos(x)
        x = tf.math.divide(
            tf.math.sin(tf.multiply(x, self.arange)),
            tf.math.sin(
                tf.multiply(0.5, x)
            )
        )

        return x


@tfk.utils.register_keras_serializable(package="arnold", name="Gegenbauer")
class Gegenbauer(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Gegenbauer polynomials.

    The Gegenbauer polynomials are generated by the three-term recurrence relation:

    * :math:`C^{\alpha}_{0}(x) = 1`
    * :math:`C^{\alpha}_{1}(x) = 2 * \alpha * x`
    * :math:`C^{\alpha}_{n+1}(x) = \frac{(2 * (n + \alpha) * x * C^{\alpha}_{n}(x)) - ((n + 2 * \alpha - 1) * C^{\alpha}_{n - 1}(x))}{n + 1}` when n >= 1

    See also: https://en.wikipedia.org/wiki/Gegenbauer_polynomials#Characterizations

    They generalize Legendre polynomials and Chebyshev polynomials, and are special cases of Jacobi polynomials.
    """

    def __init__(
            self, 
            *args,
            alpha_init:float | None = None, 
            alpha_trainable=True,
            **kwargs):
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

        :param alpha_init: Initial value for the alpha parameter of the Gegenbauer polynomials. Defaults to None (alpha with RandomNormal initialization).
        :type alpha_init: float | None = None

        :param alpha_trainable: Flag indicating whether alpha is a trainable parameter. Defaults to True
        :type alpha_trainable: bool
        """
        super().__init__(*args, **kwargs)

        self.alpha_init = alpha_init
        self.alpha_trainable = alpha_trainable

        self.alpha = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.alpha_init) if self.alpha_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='alpha',
            trainable=self.alpha_trainable
        )

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        # :math:`C^{\alpha}_{0}(x) = 1`
        gegenbauer_basis = [ tf.ones_like(x) ]

        if self.degree > 0:
            # :math:`C^{\alpha}_{1}(x) = 2 * \alpha x`
            gegenbauer_basis.append(
                2 * self.alpha * x
            )

        for n in range(2, self.degree + 1):
            # :math:`C^{\alpha}_{n+1}(x) = \frac{(2 * (n + \alpha) * x * C^{\alpha}_{n}(x)) - ((n + 2 * \alpha - 1) * C^{\alpha}_{n - 1}(x))}{n + 1}` when n >= 1
            gegenbauer_basis.append(
                ((2 * ((n - 1) + self.alpha) * x * gegenbauer_basis[n-1]) - (((n - 1) + 2 * self.alpha - 1) * gegenbauer_basis[n-2])) / n
            )
        
        return tf.stack(gegenbauer_basis, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha_init": self.alpha_init,
            "alpha_trainable": self.alpha_trainable,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="Hermite")
class Hermite(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using (physicist's) Hermite polynomials.

    The (physicist's) Hermite polynomials are generated by the three-term recurrence relation:

    * :math:`{H_{0}(x) = 1}`
    * :math:`{H_{1}(x) = 2x}`
    * :math:`{H_{n+1}(x) = 2 * x * H_{n}(x) - 2 * n * H_{n-1}(x)}` when n >= 0

    See also: https://en.wikipedia.org/wiki/Hermite_polynomials#Recurrence_relation
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        # :math:`H_{0}(x) = 1`
        hermite_basis = [ tf.ones_like(x) ]

        if self.degree > 0:
            # :math:`H_{1}(x) = x`
            hermite_basis.append(
                2.0 * x
            )

        for n in range(2, self.degree + 1):
            # :math:`{H_{n+1}(x) = 2 * x * H_{n}(x) - 2 * n * H_{n-1}(x)}` 
            hermite_basis.append(
                (2 * x * hermite_basis[n-1]) - (2 * (n - 1) * hermite_basis[n-2])
            )
        
        return tf.stack(hermite_basis, axis=-1)


@tfk.utils.register_keras_serializable(package="arnold", name="Jacobi")
class Jacobi(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Jacobi polynomials.

    The Jacobi polynomials are generated by the three-term recurrence relation:

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
    :math:`J^{\alpha, \beta}_{n}(x) = \binom{n + \alpha}{n} * _2F_1(-n, n + \alpha + \beta + 1; \alpha + 1; \frac{1-x}{2}d)
    """

    def __init__(
            self, 
            *args,
            alpha_init: float | None = None, 
            alpha_trainable=True, 
            beta_init: float | None = None, 
            beta_trainable=True,
            **kwargs):
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

        :param alpha_init: Initial value for the alpha parameter of the Jacobi polynomials. Defaults to None (a initialized to RandomNormal).
        :type alpha_init: float | None = None

        :param alpha_trainable: Flag indicating whether alpha is a trainable parameter. Defaults to True
        :type alpha_trainable: bool

        :param beta_init: Initial value for the beta parameter of the Jacobi polynomials. Defaults to None (a initialized to RandomNormal).
        :type beta_init: float | None = None

        :param beta_trainable: Flag indicating whether beta is a trainable parameter. Defaults to True
        :type beta_trainable: bool
        """ 
        super().__init__(*args, **kwargs)

        self.alpha_init = alpha_init
        self.alpha_trainable = alpha_trainable
        self.beta_init = beta_init
        self.beta_trainable = beta_trainable

        self.alpha = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.alpha_init) if self.alpha_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='alpha',
            trainable=self.alpha_trainable
        )

        self.beta = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.beta_init) if self.beta_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='beta',
            trainable=self.beta_trainable
        )

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        # See: http://lsec.cc.ac.cn/~hyu/teaching/shonm2013/STWchap3.2p.pdf (section 3.2.1.3)
        # :math:`J^{\alpha, \beta}_{0}(x) = 1`
        jacobi_basis = [ tf.ones_like(x) ]

        if self.degree > 0:
            # :math:`J^{\alpha, \beta}_{1}(x) = \frac{1}{2} * (\alpha + \beta + 2) * x + \frac{1}{2} * (\alpha - \beta)`
            jacobi_basis.append(
                (0.5 * (self.alpha - self.beta) + (self.alpha + self.beta + 2) * x / 2)
            )

        for n in range(2, self.degree + 1):
            # :math:`A^{\alpha, \beta}_{n} = \frac{(2n + \alpha + \beta +1) * (2n + \alpha + \beta + 2)} {2(n+1) * (n + \alpha + \beta + 1)}`
            A_n = tf.divide(
                (2*n + self.alpha + self.beta + 1) * (2*n + self.alpha + self.beta + 2),
                2 * (n + 1) * (n + self.alpha + self.beta + 1)
            )
            # :math:`B^{\alpha, \beta}_{n} = \frac{(\beta^{2} - \alpha^{2})(2n + \alpha + \beta +1)}{2(n+1) * (n + \alpha + \beta + 1)(2n + \alpha + \beta)}`
            B_n = tf.divide(
                (self.beta**2 - self.alpha**2) * (2 * n + self.alpha + self.beta + 1),
                2 * (n +1) * (n + self.alpha + self.beta + 1) * (2 * n + self.alpha + self.beta)
            )
            # :math:`C^{\alpha, \beta}_{n} = \frac{(n + \alpha)(n + \beta)(2n + \alpha + \beta + 2)}{(n+1) * (n + \alpha + \beta + 1)(2n + \alpha + \beta)}``
            C_n = tf.divide(
                (n + self.alpha) * (n + self.beta) * (2 * n + self.alpha + self.beta + 2),
                (n+1) * (n + self.alpha + self.beta + 1) * (2 * n + self.alpha + self.beta)
            )

            # :math:`J^{\alpha, \beta}_{n+1}(x) = (A^{\alpha, \beta}_{n} * x - B^{\alpha, \beta}_{n}) * J^{\alpha, \beta}_{n}(x) - C^{\alpha, \beta}_{n} * J^{\alpha, \beta}_{n-1}(x)` when n >= 1
            jacobi_basis.append(
                A_n * x - B_n * jacobi_basis[n-1] - C_n * jacobi_basis[n-2]
            )

        return tf.stack(jacobi_basis, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha_init": self.alpha_init,
            "alpha_trainable": self.alpha_trainable,
            "beta_init": self.beta_init,
            "beta_trainable": self.beta_trainable,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="GeneralizedLaguerre")
class GeneralizedLaguerre(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Laguerre polynomials.

    The generalized Laguerre polynomials are generated by the three-term recurrence relation:

    * :math:`L^{\alpha}_{0}(x) = 1`
    * :math:`L^{\alpha}_{1}(x) = 1 + \alpha - x`
    * :math:`L^{\alpha}_{n+1}(x) = \frac{(2n + 1 + \alpha - x) * L^{\alpha}_{n}(x) - (n + \alpha) * L^{\alpha}_{n-1}(x)}{n+1}` when n >= 1

    Special cases of the  generalized Laguerre polynomials are:
    * the Laguerre polynomials (when alpha=0); 

    See also: https://en.wikipedia.org/wiki/Laguerre_polynomials#Generalized_Laguerre_polynomials
    """

    def __init__(
            self, 
            *args,
            alpha_init: float | None = None, alpha_trainable=True, 
            **kwargs):
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

        :param alpha_init: Initial value for the alpha parameter of the GeneralizedLaguerre polynomials. Defaults to None (a initialized to RandomNormal).
        :type alpha_init: float | None = None

        :param alpha_trainable: Flag indicating whether alpha is a trainable parameter. Defaults to True
        :type alpha_trainable: bool
        """ 
        super().__init__(*args, **kwargs)

        self.alpha_init = alpha_init
        self.alpha_trainable = alpha_trainable

        self.alpha = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.alpha_init) if self.alpha_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='alpha',
            trainable=self.alpha_trainable
        )

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        # :math:`L^{\alpha}_{0}(x) = 1`
        laguerre_basis =  [tf.ones_like(x)]
        
        if self.degree > 0:
            # :math:`L^{\alpha}_{1}(x) = 1 + \alpha - x`
            laguerre_basis.append(
                1.0 + self.alpha - x
            )

        for n in range(2, self.degree + 1):
            # :math:`L^{\alpha}_{n+1}(x) = \frac{(2n + 1 + \alpha - x) * L^{\alpha}_{n}(x) - (n + \alpha) * L^{\alpha}_{n-1}(x)}{n+1}` when n >= 1
            laguerre_basis.append(
                (((2 * n - 1 + self.alpha - x) * laguerre_basis[n-1]) - ((n - 1) * self.alpha * laguerre_basis[n-2])) / n
            )

        return tf.stack(laguerre_basis, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha_init": self.alpha_init,
            "alpha_trainable": self.alpha_trainable,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="Legendre")
class Legendre(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Legendre polynomials.

    The Legendre polynomials are generated by the three-term recurrence relation:

    * :math:`P_{0}(x) = 1`
    * :math:`P_{1}(x) = x`
    * :math:`P_{n+1}(x) = \frac{(2n + 1) * x * P_{n}(x) - n * P_{n-1}(x)}{n+1}` when n >= 1

    See also: https://en.wikipedia.org/wiki/Legendre_polynomials#Definition_via_generating_function
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        # :math:`P_{0}(x) = 1`
        legendre_basis = [ tf.ones_like(x) ]
        
        if self.degree > 0:
            # :math:`P_{1}(x) = x`
            legendre_basis.append(x)

        for n in range(2, self.degree + 1):
            # :math:`P_{n+1}(x) = \frac{(2n + 1) * x * P_{n}(x) - n * P_{n-1}(x)}{n+1}` when n >= 1
            legendre_basis.append((((2 * n - 1) * x * legendre_basis[n-1]) - ((n - 1) * legendre_basis[n-2])) / n)

        return tf.stack(legendre_basis, axis=-1)


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
            *args,
            lambda_init: float | None = None, lambda_trainable=True, 
            phi_init: float | None = None, phi_trainable=True, 
            c_init: float | None = None, c_trainable=True, 
            **kwargs):
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

        :param lambda_init: Initial value for the lambda parameter of the AssociatedMeixnerPollaczek polynomials. Defaults to None (a initialized to RandomNormal).
        :type lambda_init: float | None = None

        :param lambda_trainable: Flag indicating whether lambda is a trainable parameter. Defaults to True
        :type lambda_trainable: bool

        :param phi_init: Initial value for the phi parameter of the AssociatedMeixnerPollaczek polynomials. Defaults to None (a initialized to RandomNormal).
        :type phi_init: float | None = None

        :param phi_trainable: Flag indicating whether phi is a trainable parameter. Defaults to True
        :type phi_trainable: bool

        :param c_init: Initial value for the c parameter of the AssociatedMeixnerPollaczek polynomials. Defaults to None (a initialized to RandomNormal).
        :type c_init: float | None = None

        :param c_trainable: Flag indicating whether c is a trainable parameter. Defaults to True
        :type c_trainable: bool
        """    
        super().__init__(*args, **kwargs)

        self.lambda_init = lambda_init
        self.lambda_trainable = lambda_trainable
        self.phi_init = phi_init
        self.phi_trainable = phi_trainable
        self.c_init = c_init
        self.c_trainable = c_trainable

        self.lambda_ = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.lambda_init) if self.lambda_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0),
            name='lambda',
            trainable=self.lambda_trainable
        )

        self.phi = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.phi_init) if self.phi_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0),
            name='phi',
            trainable=self.phi_trainable
        )

        self.c = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.c_init) if self.c_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0),
            name='c',
            trainable=self.c_trainable
        )

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        # :math:`P^{\lambda}_{0}(x; \phi, c)  = 1`
        meixner_pollaczek_basis = [ tf.ones_like(x) ]

        if self.degree > 0:
            # \frac{(2 * x * \sin(\phi) + 2*(n + c + \lambda)}{n + c + 1} 
            meixner_pollaczek_basis.append(
                (2 * x * tf.sin(self.phi) + 2 * (1 + self.c + self.lambda_) * tf.cos(self.phi)) / (1 + self.c + 1.0)
            )

        for n in range(2, self.degree + 1):
            # :math:`P^{\lambda}_{n+1}(x; \phi, c) = \frac{(2 * x * \sin(\phi) + 2*(n + c + \lambda)* P^{\lambda}_{n}(x; \phi, c) - (n + c + 2*\lambda - 1) * P^{\lambda}_{n-1}(x; \phi, c)}{n + c + 1}  when n >= 2
            term1 = (2 * x * tf.sin(self.phi) + 2 * (n + self.c + self.lambda_) * tf.cos(self.phi))
            term2 = (n + self.c + 2 * self.lambda_ - 1.0)
            term3 = (n + self.c + 1.0)
            meixner_pollaczek_basis.append(
                (term1 * meixner_pollaczek_basis[n-1] - term2 * meixner_pollaczek_basis[n-2]) / term3
            )
        
        return tf.stack(meixner_pollaczek_basis, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "lambda_init": self.lambda_init,
            "lambda_trainable": self.lambda_trainable,
            "phi_init": self.phi_init,
            "phi_trainable": self.phi_trainable,
            "c_init": self.c_init,
            "c_trainable": self.c_trainable,
        })
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
            *args,
            a_init: float | None = None, a_trainable=True, 
            b_init: float | None = None, b_trainable=True,
            **kwargs):
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

        :param a_init: Initial value for the a parameter of the Pollaczek polynomials. Defaults to None (a initialized to RandomNormal).
        :type a_init: float | None = None

        :param a_trainable: Flag indicating whether a is a trainable parameter. Defaults to True
        :type a_trainable: bool

        :param b_init: Initial value for the b parameter of the Pollaczek polynomials. Defaults to None (a initialized to RandomNormal).
        :type b_init: float | None = None

        :param b_trainable: Flag indicating whether b is a trainable parameter. Defaults to True
        :type b_trainable: bool
        """ 
        super().__init__(*args, **kwargs)

        self.a_init = a_init
        self.a_trainable = a_trainable
        self.b_init = b_init
        self.b_trainable = b_trainable

        self.a = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.a_init) if self.a_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='a',
            trainable=self.a_trainable
        )

        self.b = self.add_weight(
            initializer=tfk.initializers.Constant(value=self.b_init) if self.b_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='b',
            trainable=self.b_trainable
        )

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        # :math:`P_{0}(x; a, b) = 1`
        pollaczek_basis = [ tf.ones_like(x) ]
        
        if self.degree > 0:
            # :math:`P_{1}(x; a, b) = (2 * a + 1) * x + 2 * b`
            pollaczek_basis.append(
                (2 * self.a + 1) * x + 2 * self.b
            )

        for n in range(2, self.degree + 1):
            # :math:`P_{n}(x;a,b) = \frac{[(2n-1+2a)x+2b]P_{n-1}(x;a,b)-(n-1)P_{n-2}(x;a,b) }{n} when n >= 2
            pollaczek_basis.append(
                (((2 * n - 1 + 2 * self.a) * x + 2* self.b) * pollaczek_basis[n-1] - (n-1) * pollaczek_basis[n-2]) / n
            )

        return tf.stack(pollaczek_basis, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "a_init": self.a_init,
            "a_trainable": self.a_trainable,
            "b_init": self.b_init,
            "b_trainable": self.b_trainable,
        })
        return config


@tfk.utils.register_keras_serializable(package="arnold", name="Wilson")
class Wilson(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using the Wilson polynomials.

    TODO: NEEDS FIXING!
    """

    def __init__(
            self, 
            *args,
            a_init: float | None = None, a_trainable=True, 
            b_init: float | None = None, b_trainable=True,
            c_init: float | None = None, c_trainable=True, 
            d_init: float | None = None, d_trainable=True, 
            **kwargs):
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

        :param a_init: Initial value for the a parameter of the Wilson polynomials. Must be a positive, non-zero float; defaults to None (a initialized to RandomNormal).
        :type a_init: float | None = None

        :param a_trainable: Flag indicating whether a is a trainable parameter. Defaults to True
        :type a_trainable: bool

        :param b_init: Initial value for the b parameter of the Wilson polynomials. Must be a positive, non-zero float; defaults to None (a initialized to RandomNormal).
        :type b_init: float | None = None

        :param b_trainable: Flag indicating whether b is a trainable parameter. Defaults to True
        :type b_trainable: bool

        :param c_init: Initial value for the c parameter of the Wilson polynomials. Must be a positive, non-zero float; defaults to None (a initialized to RandomNormal).
        :type c_init: float | None = None

        :param c_trainable: Flag indicating whether c is a trainable parameter. Defaults to True
        :type c_trainable: bool

        :param d_init: Initial value for the parameter d of the Wilson polynomials. Must be a positive, non-zero float; defaults to None (a initialized to RandomNormal).
        :type d_init: float | None = None

        :param d_trainable: Flag indicating whether d is a trainable parameter. Defaults to True
        :type d_trainable: bool
        """ 
        super().__init__(*args, **kwargs)

        self.a_init = a_init
        self.a_trainable = a_trainable
        self.b_init = b_init
        self.b_trainable = b_trainable
        self.c_init = c_init
        self.c_trainable = c_trainable
        self.d_init = d_init
        self.d_trainable = d_trainable

        self.a = self.add_weight(
            initializer=tfk.initializers.Constant(value=tf.math.log(self.a_init)) if self.a_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='a_logits',
            trainable=self.a_trainable
        )

        self.b = self.add_weight(
            initializer=tfk.initializers.Constant(value=tf.math.log(self.b_init)) if self.b_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='b_logits',
            trainable=self.b_trainable
        )

        self.c = self.add_weight(
            initializer=tfk.initializers.Constant(value=tf.math.log(self.c_init)) if self.c_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='c_logits',
            trainable=self.c_trainable
        )

        self.d = self.add_weight(
            initializer=tfk.initializers.Constant(value=tf.math.log(self.d_init)) if self.d_init else tfk.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            name='d_logits',
            trainable=self.d_trainable
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "a_init": self.a_init,
            "a_trainable": self.a_trainable,
            "b_init": self.b_init,
            "b_trainable": self.b_trainable,
            "c_init": self.c_init,
            "c_trainable": self.c_trainable,
            "d_init": self.d_init,
            "d_trainable": self.d_trainable,
        })
        return config


    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):

        # recover parameters from logits
        a = tf.math.exp(self.a)
        b = tf.math.exp(self.b)
        c = tf.math.exp(self.c)
        d = tf.math.exp(self.d)
        n = self.degree + 1

        return tf.stack(
            list(
                map(
                    lambda degree: generalized_hypergeometric(
                        [-n * tf.ones_like(x), (a+b+c+d+n-1)* tf.ones_like(x), a* tf.ones_like(x) - x, a* tf.ones_like(x) + x], 
                        [(a+b) * tf.ones_like(x), (a+c) * tf.ones_like(x), (a+d) * tf.ones_like(x)], 
                        1.0, 
                        n + 1),
                    range(n)
                )
            ),
            axis=-1
        )
