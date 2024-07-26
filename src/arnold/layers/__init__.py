"""
Implementation of KAN layers using wavelets.
"""

from .core.wavelet.wavelet_base import WaveletBase
from .core.wavelet.ricker import Ricker
from .core.wavelet.morelet import Morelet
from .core.wavelet.meyer import Meyer
# from .core.wavelet.shannon import Shannon
# from .core.wavelet.bump import Bump
from .core.wavelet.poisson import Poisson
from .core.wavelet.dog import DerivativeOfGaussian

from .core.radial.gaussian import GaussianRBF
from .core.radial.inverse_quadratic import InverseQuadraticRBF
from .core.radial.inverse_multiquadric import InverseMultiQuadricRBF

from .core.polynomial.boubaker import Boubaker
from .core.polynomial.laurent import Laurent

from .core.polynomial.orthogonal.al_salam_carlitz import AlSalamCarlitz1st, AlSalamCarlitz2nd
from .core.polynomial.orthogonal.askey_wilson import AskeyWilson
from .core.polynomial.orthogonal.bannai_ito import BannaiIto
from .core.polynomial.orthogonal.bessel import Bessel
from .core.polynomial.orthogonal.charlier import Charlier
from .core.polynomial.orthogonal.chebyshev import Chebyshev1st, Chebyshev2nd, Chebyshev3rd, Chebyshev4th
from .core.polynomial.orthogonal.gegenbauer import Gegenbauer
from .core.polynomial.orthogonal.hermite import Hermite
from .core.polynomial.orthogonal.jacobi import Jacobi
from .core.polynomial.orthogonal.laguerre import GeneralizedLaguerre
from .core.polynomial.orthogonal.legendre import Legendre
from .core.polynomial.orthogonal.meixner_pollaczek import AssociatedMeixnerPollaczek
from .core.polynomial.orthogonal.pollaczek import Pollaczek

__all__ = [
    "Poisson",
    "Ricker",
    "Morelet",
    "Meyer",
    # "Shannon",
    # "Bump",
    "DerivativeOfGaussian",
    "GaussianRBF", 
    "InverseQuadraticRBF",
    "InverseMultiQuadricRBF",
    "Boubaker",
    "Laurent",
    "AlSalamCarlitz1st", 
    "AlSalamCarlitz2nd",
    "AskeyWilson",
    "BannaiIto",
    "Bessel",
    "Charlier",
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