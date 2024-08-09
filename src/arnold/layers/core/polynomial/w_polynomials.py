import tensorflow as tf

from .poly_base import PolynomialBase

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="Lucas")
class Lucas(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Lucas polynomials.

    The Lucas polynomials are the w-polynomials obtained by setting p(x)=x and q(x)=1 in the Lucas polynomial sequence:

    * :math:`w_{0}(x) = L_{0}(x) = 2`
    * :math:`w_{1}(x) = L_{0}(x) = x`
    * :math:`w_{n+1}(x) = p(x) * w_{n}(x) + q(x) * w_{n-1}(x)` when n >= 1


    It is given explicitly by

    :math:`L_{n}(x) = 2^{-n} * ( (x - \sqrt(x^{2} + 4) )^{n} + (x + \sqrt(x^{2} + 4) )^{n} )`

    See also: https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        # :math:`L_{0}(x) = 2`
        lucas_basis = [ 2 * tf.ones_like(x) ]
        
        if self.degree > 0:
            # :math:`L_{1}(x) = x`
            lucas_basis.append(x)

        for n in range(2, self.degree + 1):
            # :math:`L_{n+1}(x) = x * L_{n}(x) + L_{n-1}(x)` when n >= 1
            lucas_basis.append(
                x * lucas_basis[n-1] + lucas_basis[n-2]
            )
        
        return tf.stack(lucas_basis, axis=-1)


@tfk.utils.register_keras_serializable(package="arnold", name="FermatLucas")
class FermatLucas(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Fermat polynomials.

    The Fermat-Lucas polynomials are the w-polynomials obtained by setting p(x)=3*x and q(x)=-2 in the Lucas polynomial sequence:

    * :math:`w_{0}(x) = f_{0}(x) = 2`
    * :math:`w_{1}(x) = f_{1}(x) = 3*x`
    * :math:`w_{n+1}(x) = p(x) * w_{n}(x) + q(x) * w_{n-1}(x)` when n >= 1

    See also: https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        # :math:`f_{0}(x) = 2`
        fermat_lucas_basis = [ 2 * tf.ones_like(x) ]
        
        if self.degree > 0:
            # :math:`f_{1}(x) = 3 * x`
            fermat_lucas_basis.append( 3 * x)

        for n in range(2, self.degree + 1):
            # :math:`f_{n+1}(x) = 3 * x * f_{n}(x) - 2 * f_{n-1}(x)` when n >= 1
            fermat_lucas_basis.append(
                3 * x * fermat_lucas_basis[n-1] - 2 * fermat_lucas_basis[n-2]
            )
        
        return tf.stack(fermat_lucas_basis, axis=-1)


@tfk.utils.register_keras_serializable(package="arnold", name="Fermat")
class Fermat(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Fermat polynomials.

    The Fermat polynomials are the w-polynomials obtained by setting p(x)=3*x and q(x)=-2 in the Lucas polynomial sequence:

    * :math:`w_{0}(x) = F_{0}(x) = 0`
    * :math:`w_{1}(x) = F_{1}(x) = 1`
    * :math:`w_{n+1}(x) = p(x) * w_{n}(x) + q(x) * w_{n-1}(x)` when n >= 1

    See also: https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        # :math:`F_{0}(x) = 0`
        fermat_basis = [ tf.zeros_like(x) ]
        
        if self.degree > 0:
            # :math:`F_{1}(x) = 1`
            fermat_basis.append(tf.ones_like(x))

        for n in range(2, self.degree + 1):
            # :math:`F_{n+1}(x) = 3 * x * F_{n}(x) - 2 * F_{n-1}(x)` when n >= 1
            fermat_basis.append(
                3 * x * fermat_basis[n-1] - 2 * fermat_basis[n-2]
            )
        
        return tf.stack(fermat_basis, axis=-1)


@tfk.utils.register_keras_serializable(package="arnold", name="JacobsthalLucas")
class JacobsthalLucas(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Jacobsthal-Lucas polynomials.

    The Jacobsthal-lucas polynomials are the w-polynomials obtained by setting p(x)=1 and q(x)=2*x in the Lucas polynomial sequence:

    * :math:`w_{0}(x) = j_{0}(x) = 2`
    * :math:`w_{1}(x) = j_{1}(x) = 1`
    * :math:`w_{n+1}(x) = p(x) * w_{n}(x) + q(x) * w_{n-1}(x)` when n >= 1

    See also: https://www.fq.math.ca/Scanned/35-2/horadam.pdf
    See also: https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        # :math:`j_{0}(x) = 2`
        jacobsthal_lucas_basis = [ 2 * tf.ones_like(x) ]
        
        if self.degree > 0:
            # :math:`j_{1}(x) = x`
            jacobsthal_lucas_basis.append(tf.ones_like(x))

        for n in range(2, self.degree + 1):
            # :math:`j_{n+1}(x) = 1 * j_{n}(x) + 2 * x * j_{n-1}(x)` when n >= 1
            jacobsthal_lucas_basis.append(
                jacobsthal_lucas_basis[n-1] + 2 * x * jacobsthal_lucas_basis[n-2]
            )
        
        return tf.stack(jacobsthal_lucas_basis, axis=-1)


@tfk.utils.register_keras_serializable(package="arnold", name="Jacobsthal")
class Jacobsthal(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Jacobsthal polynomials.

    The Jacobsthal polynomials are the w-polynomials obtained by setting p(x)=1 and q(x)=2*x in the Lucas polynomial sequence:

    * :math:`w_{0}(x) = J_{0}(x) = 0`
    * :math:`w_{1}(x) = J_{1}(x) = 1`
    * :math:`w_{n+1}(x) = p(x) * w_{n}(x) + q(x) * w_{n-1}(x)` when n >= 1

    See also: https://www.fq.math.ca/Scanned/35-2/horadam.pdf
    See also: https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        # :math:`J_{0}(x) = 2`
        jacobsthal_basis = [ tf.zeros_like(x) ]
        
        if self.degree > 0:
            # :math:`J_{1}(x) = x`
            jacobsthal_basis.append(tf.ones_like(x))

        for n in range(2, self.degree + 1):
            # :math:`J_{n+1}(x) = 1 * J_{n}(x) + 2 * x * J_{n-1}(x)` when n >= 1
            jacobsthal_basis.append(
                jacobsthal_basis[n-1] + 2 * x * jacobsthal_basis[n-2]
            )
        
        return tf.stack(jacobsthal_basis, axis=-1)


@tfk.utils.register_keras_serializable(package="arnold", name="PellLucas")
class PellLucas(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using Pell-Lucas polynomials.

    The Pell-Lucas polynomials are the w-polynomials obtained by setting p(x)=2*x and q(x)=1 in the Lucas polynomial sequence:

    * :math:`w_{0}(x) = Q_{0}(x) = 2`
    * :math:`w_{1}(x) = Q_{0}(x) = 2x`
    * :math:`w_{n+1}(x) = p(x) * w_{n}(x) + q(x) * w_{n-1}(x)` when n >= 1

    See also: https://www.mathstat.dal.ca/FQ/Scanned/23-1/horadam.pdf
    See also: https://mathworld.wolfram.com/LucasPolynomialSequence.html
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        # :math:`Q_{0}(x) = 2`
        pell_lucas_basis = [ 2 * tf.ones_like(x) ]
        
        if self.degree > 0:
            # :math:`Q_{1}(x) = x`
            pell_lucas_basis.append(2 * x)

        for n in range(2, self.degree + 1):
            # :math:`Q_{n+1}(x) = x * Q_{n}(x) + Q_{n-1}(x)` when n >= 1
            pell_lucas_basis.append(
                2 * x * pell_lucas_basis[n-1] + pell_lucas_basis[n-2]
            )
        
        return tf.stack(pell_lucas_basis, axis=-1)


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

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def pseudo_vandermonde(self, x):
        # :math:`P_{0}(x) = 0`
        pell_basis = [ tf.zeros_like(x) ]
        
        if self.degree > 0:
            # :math:`P_{1}(x) = x`
            pell_basis.append( tf.ones_like(x) )

        for n in range(2, self.degree + 1):
            # :math:`P_{n+1}(x) = 2 * x * P_{n}(x) + P_{n-1}(x)` when n >= 1
            pell_basis.append(
                2 * x * pell_basis[n-1] + pell_basis[n-2]
            )
        
        return tf.stack(pell_basis, axis=-1)
