# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Radial Basis Function (RBF) KAN Layers
======================================

This package provides Kolmogorov-Arnold Network layers using radial basis functions.

RBF kernels compute :math:`\phi(r)` where :math:`r = \|x - \mu_k\|` is the distance
to grid centers. Common kernels include:

- **Gaussian**: :math:`\phi(r) = \exp(-(\varepsilon r)^2)` — localized, smooth
- **Multiquadric**: :math:`\phi(r) = \sqrt{1 + (\varepsilon r)^2}` — global, unbounded
- **Inverse Multiquadric**: :math:`\phi(r) = 1/\sqrt{1 + (\varepsilon r)^2}` — decays smoothly
- **Thin Plate Spline**: :math:`\phi(r) = r^2 \ln(r)` — minimizes bending energy
- **Cauchy**: :math:`\phi(r) = 1/(1 + (r/\sigma)^2)` — heavy-tailed decay

All shape parameters (:math:`\varepsilon`, :math:`\sigma`, etc.) are stored in logits
and transformed via ``softplus`` to ensure positivity and stable gradients.

Classes
-------
.. autosummary::
   :toctree: generated/

   RBFBase
   GaussianRBF
   MultiquadricRBF
   InverseMultiQuadricRBF
   ThinPlateSplineRBF
   CauchyRBF
   LinearRBF
   CubicRBF
   PowerRBF
   InverseQuadricRBF
   ExponentialRBF

Examples
--------
>>> from arnold.layers.core.rbf import GaussianRBF
>>> layer = GaussianRBF(units=32, num_grids=16)
>>> output = layer(inputs)

See Also
--------
arnold.layers.core.polynomial : Polynomial basis KAN layers
arnold.layers.core.wavelets : Wavelet basis KAN layers
"""

from arnold.layers.core.rbf.base import RBFBase
from arnold.layers.core.rbf.gaussian import GaussianRBF
from arnold.layers.core.rbf.multiquadric import MultiquadricRBF
from arnold.layers.core.rbf.inverse_multiquadric import InverseMultiQuadricRBF
from arnold.layers.core.rbf.thin_plate_spline import ThinPlateSplineRBF
from arnold.layers.core.rbf.cauchy import CauchyRBF
from arnold.layers.core.rbf.linear import LinearRBF
from arnold.layers.core.rbf.cubic import CubicRBF
from arnold.layers.core.rbf.power import PowerRBF
from arnold.layers.core.rbf.inverse_quadric import InverseQuadricRBF
from arnold.layers.core.rbf.exponential import ExponentialRBF

__all__ = [
    # Base
    "RBFBase",
    # Standard RBFs
    "GaussianRBF",
    "MultiquadricRBF",
    "InverseMultiQuadricRBF",
    "ThinPlateSplineRBF",
    "CauchyRBF",
    # Simple RBFs
    "LinearRBF",
    "CubicRBF",
    "PowerRBF",
    "InverseQuadricRBF",
    "ExponentialRBF",
]
