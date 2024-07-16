"""
Implementation of KAN layers using wavelets.
"""

from .wavelet_base import WaveletBase
from .ricker import Ricker
from .morelet import Morelet
from .meyer import Meyer
from .shannon import Shannon
from .bump import Bump
from .poisson import Poisson

__all__ = [
    "Poisson",
    "Ricker",
    "Morelet",
    "Meyer",
    "Shannon",
    "Bump"
]