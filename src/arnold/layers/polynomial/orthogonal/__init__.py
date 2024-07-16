"""
Implementation of KAN layers using orthogonal polynomial bases.
"""

from .al_salam_carlitz import AlSalamCarlitz1st, AlSalamCarlitz2nd
from .askey_wilson import AskeyWilson
from .bannai_ito import BannaiIto
from .bessel import Bessel
from .chebyshev import Chebyshev1st, Chebyshev2nd, Chebyshev3rd, Chebyshev4th
from .gegenbauer import Gegenbauer
from .hermite import Hermite
from .jacobi import Jacobi
from .laguerre import GeneralizedLaguerre
from .legendre import Legendre
from .meixner_pollaczek import AssociatedMeixnerPollaczek
from .pollaczek import Pollaczek


__all__ = [
    "AlSalamCarlitz1st", 
    "AlSalamCarlitz2nd",
    "AskeyWilson",
    "BannaiIto",
    "Bessel",
    "Chebyshev1st", 
    "Chebyshev2nd",
    "Chebyshev3rd", 
    "Chebyshev4th",
    "Gegenbauer", 
    "Hermite",
    "Jacobi",
    "GeneralizedLaguerre",
    "Legendre",
    "AssociatedMeixnerPollaczek",
    "Pollaczek"
]
