"""
Implementation of KAN layers using w-polynomial bases.
"""

from .fermat_lucas import FermatLucas
from .fermat import Fermat
from  arnold.layers.polynomial.fibonacci.fibonacci import Fibonacci
from .jacobsthal_lucas import JacobsthalLucas
from .jacobsthal import Jacobsthal
from .lucas import Lucas
from .pell_lucas import PellLucas
from .pell import Pell

from arnold.layers.polynomial.orthogonal.chebyshev import Chebyshev1st, Chebyshev2nd 


__all__ = [
    "FermatLucas", 
    "Fermat",
    "Fibonacci",
    "JacobsthalLucas",
    "Jacobsthal",
    "Lucas", 
    "PellLucas",
    "Pell", 
    "Chebyshev1st",
    "Chebyshev2nd",
]
