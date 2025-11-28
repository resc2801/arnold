# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Wavelet-based KAN layers.

This package provides wavelet layers for Kolmogorov-Arnold Networks,
including continuous wavelets (Ricker, Morelet, Meyer) and discrete
filter-bank wavelets (Haar, Daubechies, Symlet, Coiflet).
"""
# Import coefficients first (no circular dependency)
from arnold.layers.core.wavelets.coefficients import (
    COIFLET_COEFFICIENTS,
    DAUBECHIES_COEFFICIENTS,
    SYMLET_COEFFICIENTS,
)

# Then import classes that may use the coefficients
from arnold.layers.core.wavelets.base import WaveletBase
from arnold.layers.core.wavelets.bump import Bump
from arnold.layers.core.wavelets.coiflet import Coiflet
from arnold.layers.core.wavelets.daubechies import Daubechies
from arnold.layers.core.wavelets.derivative_of_gaussian import DerivativeOfGaussian
from arnold.layers.core.wavelets.haar import Haar
from arnold.layers.core.wavelets.meyer import Meyer
from arnold.layers.core.wavelets.morelet import Morelet
from arnold.layers.core.wavelets.poisson import Poisson
from arnold.layers.core.wavelets.ricker import Ricker
from arnold.layers.core.wavelets.shannon import Shannon
from arnold.layers.core.wavelets.symlet import Symlet


__all__ = [
    # Base
    "WaveletBase",
    # Continuous wavelets
    "Bump",
    "DerivativeOfGaussian",
    "Meyer",
    "Morelet",
    "Poisson",
    "Ricker",
    "Shannon",
    # Discrete filter-bank wavelets
    "Haar",
    "Daubechies",
    "Symlet",
    "Coiflet",
    # Coefficients
    "DAUBECHIES_COEFFICIENTS",
    "SYMLET_COEFFICIENTS",
    "COIFLET_COEFFICIENTS",
]
