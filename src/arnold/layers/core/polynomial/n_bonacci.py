import tensorflow as tf
from .poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers

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

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        # :math:`F_{0}(x) = 0`
        # :math:`F_{1}(x) = 1`
        fibonacci_basis = [ 
            tf.zeros_like(x),
            tf.ones_like(x)
        ]

        for n in range(2, self.degree + 1):
            # :math:`F_{n+1}(x) = x * F_{n}(x) + F_{n-1}(x)` when n >= 1
            fibonacci_basis.append(
                tf.math.add_n(
                    x * fibonacci_basis[n-1],
                    fibonacci_basis[n-2]
                )
            )

        return tf.stack(fibonacci_basis[0:(self.degree + 1)], axis=-1)


@tfk.utils.register_keras_serializable(package="arnold", name="Heptanacci")
class Heptanacci(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Heptanacci polynomials.

    Heptanacci polynomials are a generalization of the Fibonacci polynomials.
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        heptanacci_basis = [ 
            tf.zeros_like(x),
            tf.ones_like(x),
            x,
            x,
            x**2,
            x**2,
            x**3
        ]

        for n in range(7, self.degree + 1):
            heptanacci_basis.append(
                tf.math.add_n([
                    x * heptanacci_basis[n-1],
                    heptanacci_basis[n-2], 
                    heptanacci_basis[n-3],
                    heptanacci_basis[n-4],
                    heptanacci_basis[n-5],
                    heptanacci_basis[n-6],
                    heptanacci_basis[n-7]
                ])
            )

        return tf.stack(heptanacci_basis[0:(self.degree + 1)], axis=-1)


@tfk.utils.register_keras_serializable(package="arnold", name="Hexanacci")
class Hexanacci(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Heptanacci polynomials.

    Hexanacci polynomials are a generalization of the Fibonacci polynomials.
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        # :math:`_{0}(x) = 0`
        hexanacci_basis = [ 
            tf.zeros_like(x),
            tf.ones_like(x),
            x,
            x**2,
            x**2,
            x**3,
        ]

        for n in range(6, self.degree + 1):
            # :math:`F_{n+1}(x) = x * F_{n}(x) + F_{n-1}(x)` when n >= 1
            hexanacci_basis.append(
                tf.math.add_n([
                    x * hexanacci_basis[n-1],
                    hexanacci_basis[n-2], 
                    hexanacci_basis[n-3],
                    hexanacci_basis[n-4],
                    hexanacci_basis[n-5],
                    hexanacci_basis[n-6],
                ])
            )

        return tf.stack(hexanacci_basis[0:(self.degree + 1)], axis=-1)


@tfk.utils.register_keras_serializable(package="arnold", name="Octanacci")
class Octanacci(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Heptanacci polynomials.

    Octanacci polynomials are a generalization of the Fibonacci polynomials.
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        octanacci_basis = [ 
            tf.zeros_like(x),
            tf.ones_like(x),
            x,
            x**2,
            x**2,
            x**3,
            x**3,
            x**4
        ]

        for n in range(8, self.degree + 1):
            octanacci_basis.append(
                tf.math.add_n([
                    x * octanacci_basis[n-1],
                    octanacci_basis[n-2], 
                    octanacci_basis[n-3],
                    octanacci_basis[n-4],
                    octanacci_basis[n-5],
                    octanacci_basis[n-6],
                    octanacci_basis[n-7],
                    octanacci_basis[n-8]
                ])
            )

        return tf.stack(octanacci_basis[0:(self.degree + 1)], axis=-1)


@tfk.utils.register_keras_serializable(package="arnold", name="Pentanacci")
class Pentanacci(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Heptanacci polynomials.

    Pentanacci polynomials are a generalization of the Fibonacci polynomials.
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        pentanacci_basis = [ 
            tf.zeros_like(x),
            tf.ones_like(x),
            x,
            x,
            x**2,
        ]

        for n in range(5, self.degree + 1):
            pentanacci_basis.append(
                tf.math.add_n([
                    x * pentanacci_basis[n-1],
                    pentanacci_basis[n-2], 
                    pentanacci_basis[n-3],
                    pentanacci_basis[n-4],
                    pentanacci_basis[n-5],
                ])
            )

        return tf.stack(pentanacci_basis[0:(self.degree + 1)], axis=-1)


@tfk.utils.register_keras_serializable(package="arnold", name="Tetranacci")
class Tetranacci(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Heptanacci polynomials.

    Tetranacci polynomials are a generalization of the Fibonacci polynomials.
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        tetranacci_basis = [ 
            tf.zeros_like(x),
            tf.ones_like(x),
            x,
            x**2,
        ]

        for n in range(4, self.degree + 1):
            tetranacci_basis.append(
                tf.math.add_n([
                    x * tetranacci_basis[n-1],
                    tetranacci_basis[n-2], 
                    tetranacci_basis[n-3],
                    tetranacci_basis[n-4]
                ])
            )

        return tf.stack(tetranacci_basis[0:(self.degree + 1)], axis=-1)

