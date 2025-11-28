# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Orthogonal polynomial basis KAN layers.

This module provides a comprehensive suite of orthogonal polynomial families
for Kolmogorov-Arnold Network layers, organized by mathematical lineage.

Family Hierarchy
----------------
The families are organized as follows:

**Jacobi Family** (classical, continuous orthogonality on [-1, 1] or variants):
    - :class:`Jacobi` — General Jacobi polynomials :math:`P_n^{(\alpha,\beta)}(x)`
    - :class:`Legendre` — Jacobi with :math:`\alpha = \beta = 0`
    - :class:`Gegenbauer` — Jacobi with :math:`\alpha = \beta`
    - :class:`Chebyshev` — Abstract base for Chebyshev polynomials
    - :class:`Chebyshev1st` — Chebyshev of the first kind :math:`T_n(x)`
    - :class:`Chebyshev2nd` — Chebyshev of the second kind :math:`U_n(x)`
    - :class:`Chebyshev3rd` — Chebyshev of the third kind :math:`V_n(x)`
    - :class:`Chebyshev4th` — Chebyshev of the fourth kind :math:`W_n(x)`

**Laguerre Family** (orthogonal on :math:`[0, \infty)` with exponential weight):
    - :class:`GeneralizedLaguerre` — :math:`L_n^{(\alpha)}(x)`

**Hermite Family** (orthogonal on :math:`(-\infty, \infty)` with Gaussian weight):
    - :class:`Hermite` — Physicist's :math:`H_n(x)` or Probabilist's :math:`He_n(x)`

**Bessel Family** (orthogonal on the unit circle):
    - :class:`Bessel` — Bessel polynomials :math:`y_n(x)`

**Meixner-Pollaczek Family** (continuous Hahn-type):
    - :class:`AssociatedMeixnerPollaczek` — :math:`P_n^{(\lambda)}(x; \phi)`
    - :class:`Pollaczek` — Classical Pollaczek polynomials

**Wilson Family** (continuous orthogonality on :math:`(0, \infty)`):
    - :class:`Wilson` — Wilson polynomials :math:`W_n(x^2; a, b, c, d)`

**q-Orthogonal (Al-Salam-Carlitz):**
    - :class:`AlSalamCarlitz` — Abstract base for Al-Salam-Carlitz polynomials
    - :class:`AlSalamCarlitz1st` — Al-Salam-Carlitz I
    - :class:`AlSalamCarlitz2nd` — Al-Salam-Carlitz II

**q-Orthogonal (Askey-Wilson):**
    - :class:`AskeyWilson` — Top of the q-Askey scheme

**Discrete Orthogonal:**
    - :class:`BannaiIto` — Bannai-Ito polynomials
    - :class:`Charlier` — Charlier polynomials (Poisson weight)

Example
-------
>>> from arnold.layers.core.polynomial.orthogonal import Legendre, Chebyshev1st
>>> layer = Legendre(degree=5, units=32)
>>> chebyshev_layer = Chebyshev1st(degree=8, units=64, orthonormal=True)

See Also
--------
arnold.layers.core.polynomial.non_orthogonal : Non-orthogonal polynomial families
arnold.layers.core.polynomial.hypergeometric : Hypergeometric-based polynomials
"""

# Jacobi family
from .jacobi_family import (
    Chebyshev,
    Chebyshev1st,
    Chebyshev2nd,
    Chebyshev3rd,
    Chebyshev4th,
    Gegenbauer,
    Jacobi,
    Legendre,
)

# Laguerre family
from .laguerre_family import GeneralizedLaguerre

# Hermite family
from .hermite_family import Hermite

# Bessel family
from .bessel_family import Bessel

# Meixner-Pollaczek family
from .meixner_pollaczek import AssociatedMeixnerPollaczek
from .pollaczek import Pollaczek

# Wilson family
from .wilson import Wilson

# Al-Salam-Carlitz (q-orthogonal)
from .al_salam_carlitz import AlSalamCarlitz, AlSalamCarlitz1st, AlSalamCarlitz2nd

# Askey-Wilson (q-orthogonal)
from .askey_wilson import AskeyWilson

# Discrete orthogonal
from .bannai_ito import BannaiIto
from .charlier import Charlier


__all__ = [
    # Jacobi family
    "Jacobi",
    "Legendre",
    "Gegenbauer",
    "Chebyshev",
    "Chebyshev1st",
    "Chebyshev2nd",
    "Chebyshev3rd",
    "Chebyshev4th",
    # Laguerre family
    "GeneralizedLaguerre",
    # Hermite family
    "Hermite",
    # Bessel family
    "Bessel",
    # Meixner-Pollaczek family
    "AssociatedMeixnerPollaczek",
    "Pollaczek",
    # Wilson family
    "Wilson",
    # Al-Salam-Carlitz (q-orthogonal)
    "AlSalamCarlitz",
    "AlSalamCarlitz1st",
    "AlSalamCarlitz2nd",
    # Askey-Wilson (q-orthogonal)
    "AskeyWilson",
    # Discrete orthogonal
    "BannaiIto",
    "Charlier",
]
