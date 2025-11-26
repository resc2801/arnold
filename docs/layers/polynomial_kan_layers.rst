Polynomial KAN layers
******************************************

:math:`\texttt{ARNOLD}` offers a wide range of KAN layers with different polynomial bases all of which are subclassed from 

.. autoclass:: arnold.layers.core.polynomial.PolynomialBase
    :members: __init__

In case you need a KAN layer with some special polynomial basis, the easiest way would be to implement your very own

.. automethod:: arnold.layers.core.polynomial.PolynomialBase.pseudo_vandermonde


Orthogonal polynomial bases
==========================================

Orthogonal polynomials, such as Legendre, Chebyshev, and Hermite polynomials, have coefficients that are uncorrelated when integrated over a certain range with a specific weight function. 
The orthogonality condition ensures that each polynomial basis function captures unique aspects of the data, minimizing overlap and improving the network's learning efficiency.

.. autoclass:: arnold.layers.core.polynomial.orthogonal.AlSalamCarlitz1st
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.orthogonal.AlSalamCarlitz2nd
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.orthogonal.AskeyWilson
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.orthogonal.BannaiIto
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Bessel
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Charlier
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Chebyshev1st
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Chebyshev2nd
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Chebyshev3rd
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Chebyshev4th
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Gegenbauer
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Hermite
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Jacobi
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.orthogonal.GeneralizedLaguerre
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Legendre
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.orthogonal.AssociatedMeixnerPollaczek
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Pollaczek
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Wilson
    :members: __init__
    

Non-orthogonal polynomial bases
==========================================

Non-orthogonal polynomials do not satisfy the condition of orthogonality, meaning their inner product is not necessarily zero for distinct polynomials. 
This may allow for a broader selection of basis functions, which can be tailored to fit particular types of data or specific functional forms that might be more challenging to capture with orthogonal polynomials.


.. autoclass:: arnold.layers.core.polynomial.Boubaker
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.Laurent
    :members: __init__


Lucas polynomial sequences
===============================

A Lucas polynomial sequence is a pair of generalized polynomials which generalize the Lucas sequence to polynomials. 
:math:`\mathtt{ARNOLD}` offers a number of special cases.

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Chebyshev1st
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Chebyshev2nd
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.w_polynomials.Fermat
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.w_polynomials.FermatLucas
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.fibonacci.Fibonacci
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.w_polynomials.Jacobsthal
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.w_polynomials.JacobsthalLucas
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.w_polynomials.FermatLucas
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.w_polynomials.Pell
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.w_polynomials.PellLucas
    :members: __init__


(Generalized) Fibonacci polynomials
==========================================

The Fibonacci polynomials are a polynomial sequence which can be considered as a generalization of the Fibonacci numbers. 
:math:`\mathtt{ARNOLD}`  also provides polynomial sequences based on Fibonacci numbers of higher order.

.. autoclass:: arnold.layers.core.polynomial.fibonacci.Fibonacci
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.fibonacci.Tetranacci
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.fibonacci.Pentanacci
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.fibonacci.Hexanacci
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.fibonacci.Heptanacci
    :members: __init__

.. autoclass:: arnold.layers.core.polynomial.fibonacci.Octanacci
    :members: __init__
