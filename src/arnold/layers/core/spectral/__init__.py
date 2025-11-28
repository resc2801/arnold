# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Spectral-based KAN layers.

This package provides spectral layers for Kolmogorov-Arnold Networks
using frequency-domain representations.

.. note::
    Most layers are stubs (Phase 9a). Full implementation in Phase 9b.

Available layers:
- :class:`FourierKAN`: Trigonometric basis (stub)
- :class:`RandomFourierFeatures`: RFF kernel approximation (stub)
- :class:`CosineBasis`: DCT basis (stub)
- :class:`SincBasis`: Shannon sampling (stub)
- :class:`WindowedSinc`: Windowed sinc (stub)
- :class:`Lorentzian`: Cauchy spectral basis (stub)
- :class:`DiracComb`: Impulse train (stub)
"""
from arnold.layers.core.spectral.base import SpectralBase
from arnold.layers.core.spectral.cosine_basis import CosineBasis
from arnold.layers.core.spectral.dirac_comb import DiracComb
from arnold.layers.core.spectral.fourier_basis import FourierKAN
from arnold.layers.core.spectral.lorentzian import Lorentzian
from arnold.layers.core.spectral.random_fourier_features import RandomFourierFeatures
from arnold.layers.core.spectral.sinc import SincBasis
from arnold.layers.core.spectral.windowed_sinc import WindowedSinc


__all__ = [
    "SpectralBase",
    "FourierKAN",
    "RandomFourierFeatures",
    "CosineBasis",
    "SincBasis",
    "WindowedSinc",
    "Lorentzian",
    "DiracComb",
]
