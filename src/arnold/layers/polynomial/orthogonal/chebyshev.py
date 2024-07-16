from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
from sympy.polys.orthopolys import chebyshevt_poly

from arnold.layers.polynomial.poly_base import PolynomialBase

from sympy.polys.polytools import poly
from sympy.polys.orthopolys import chebyshevt_poly
from sympy import var

tfk = tf.keras
tfkl = tfk.layers


class Chebyshev(PolynomialBase, ABC):
    """
    Abstract base class for Kolmogorov-Arnold Network layer using Chebyshev polynomial basis.

    TODO: check https://www.mathematik.uni-kassel.de/~koepf/Publikationen/cheby.pdf
    """
    def __init__(
            self, 
            *args,
            **kwargs):
        
        super().__init__(*args, **kwargs)
        self.arange = tf.constant(tf.range(0, self.degree + 1, 1, dtype=float))


class Chebyshev1st(Chebyshev):
    """
    Kolmogorov-Arnold Network layer using 1st kind Chebyshev polynomials 
    in trigonometric formulation 
        :math:`T_{n}(x) = \cos(n * \arccos(x))` 
    for |x|≤1.

    See: https://core.ac.uk/download/pdf/82763706.pdf   
    """

    @tf.function
    def poly_basis(self, x):
        # Reshape to (batch_size, input_dim, self.degree + 1)
        x = tf.reshape(x, (-1, self.input_dim, 1))
        x = tf.tile(x, (1, 1, self.degree + 1))
        
        # Prepare :math:`T_{n}(x) = \cos(n * \arccos(x))` 
        x = tf.math.acos(x)
        x = tf.multiply(x, self.arange)
        x = tf.math.cos(x)

        return x
        

class Chebyshev2nd(Chebyshev):
    """
    Kolmogorov-Arnold Network layer using 2nd kind Chebyshev polynomials 
    in trigonometric formulation 
        :math:`U_{n}(x) := `\frac{\sin((n+1) * \arccos(x))}{\sin(\arccos(x))}` 
    for |x|≤1.

    See: https://core.ac.uk/download/pdf/82763706.pdf
    """

    @tf.function
    def poly_basis(self, x):
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


class Chebyshev3rd(Chebyshev):
    """
    Kolmogorov-Arnold Network layer using 3rd kind Chebyshev polynomials 
    in trigonometric formulation 
        :math:`V_{n}(x) := `\frac{\cos((n + \tfrac{1}{2}) * \arccos(x))}{\cos(\tfrac{1}{2} * \arccos(x))}` 
    for |x|≤1.

    See: https://core.ac.uk/download/pdf/82763706.pdf
    """

    @tf.function
    def poly_basis(self, x):
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
    

class Chebyshev4th(Chebyshev):
    """
    Kolmogorov-Arnold Network layer using 4th kind Chebyshev polynomials 
    in trigonometric formulation 
        :math:`W_{n}(x) := `\frac{\sin((n + \tfrac{1}{2}) * \arccos(x))}{\sin(\tfrac{1}{2} * \arccos(x))}` 
    for |x|≤1.

    See: https://core.ac.uk/download/pdf/82763706.pdf
    """

    @tf.function
    def poly_basis(self, x):
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
