from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

from arnold.layers.polynomial.poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="AlSalamCarlitz")
class AlSalamCarlitz(PolynomialBase, ABC):
    """
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
        """
        This function does something.

        :param name: The name to use.
        :type name: str.
        :param state: Current state to be in.
        :type state: bool.
        :returns:  int -- the return code.
        :raises: AttributeError, KeyError
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
    """
    Kolmogorov-Arnold Network layer using the Al-Salam-Carlitz polynomials $U^{(a)}_{n} (x;q)$.

    These polynomials satisfy the three-term recurrence relation 

    * $U^{(a)}_{-1} (x;q) = 0$
    * $U^{(a)}_{0} (x;q) = 1$
    * $U^{(a)}_{n+1} (x;q) = (x - (1 + a) * q^{n}) * U^{(a)}_{n} (x;q) + a*q^{n-1} * (1 - q^{n}) * U^{(a)}_{n-1} (x;q)$

    See also: https://core.ac.uk/download/pdf/82826366.pdf
    """

    @tf.function
    def poly_basis(self, x):
        
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
    """
    Kolmogorov-Arnold Network layer using the Al-Salam-Carlitz polynomials $V^{(a)}_{n} (x;q)$.

    There is a straightforward relationship between $U^{(a)}_{n} (x;q)$ and $V^{(a)}_{n} (x;q)$:

    * $U^{(a)}_{n} (x; 1/q) = V^{(a)}_{n} (x;q)

    See: Chihara, T.S. An Introduction to Orthogonal Polynomials; Mathematics Applied Series 13; Gordon and Breach: New York, NY, USA, 1978. Chapter VI, §10, pp. 195–198
    """

    @tf.function 
    def poly_basis(self, x):

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
