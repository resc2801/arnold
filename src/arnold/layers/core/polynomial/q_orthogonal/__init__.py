# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
q-Orthogonal polynomial KAN layers.

This package provides Kolmogorov-Arnold Network layers using q-deformed
orthogonal polynomials as basis functions. These are q-analogs of classical
orthogonal polynomials with an additional base parameter q ∈ (0, 1).

Mathematical Background
-----------------------
The q-orthogonal polynomials form a hierarchy related to the Askey scheme,
with the classical orthogonal polynomials recovered in the limit q → 1.

Key families include:

- **Askey-Wilson class**: q-Racah, q-Hahn, Dual q-Hahn, etc.
- **q-Hahn class**: Big/Little q-Jacobi, q-Krawtchouk, etc.
- **q-Hermite variants**: Discrete q-Hermite I/II, Continuous q-Hermite
- **Continuous q-polynomials**: q-Jacobi, q-Ultraspherical, q-Legendre, q-Laguerre

All layers inherit from :class:`QPolynomialBase` which provides:

- Trainable q parameter with sigmoid constraint to (0, 1)
- Three-term recurrence evaluation in float64 precision
- Optional input clipping for domain control

Classes
-------
.. autosummary::
    :toctree: generated/

    QPolynomialBase
    QHahn
    BigQJacobi
    LittleQJacobi
    QMeixner
    QKrawtchouk
    QCharlier
    QRacah
    DualQHahn
    DualQKrawtchouk
    AffineQKrawtchouk
    DiscreteQHermite1
    DiscreteQHermite2
    ContinuousQHermite
    ContinuousQJacobi
    ContinuousQUltraspherical
    QuantumQKrawtchouk
    ContinuousQLaguerre
    ContinuousQLegendre

References
----------
.. [1] Koekoek, Lesky & Swarttouw (2010). "Hypergeometric Orthogonal Polynomials
       and Their q-Analogues", Springer Monographs in Mathematics.
.. [2] NIST Digital Library of Mathematical Functions, Chapter 18
"""

from .affine_q_krawtchouk import AffineQKrawtchouk
from .base import QPolynomialBase
from .big_q_jacobi import BigQJacobi
from .continuous_q_hermite import ContinuousQHermite
from .continuous_q_jacobi import ContinuousQJacobi
from .continuous_q_laguerre import ContinuousQLaguerre
from .continuous_q_legendre import ContinuousQLegendre
from .continuous_q_ultraspherical import ContinuousQUltraspherical
from .discrete_q_hermite1 import DiscreteQHermite1
from .discrete_q_hermite2 import DiscreteQHermite2
from .dual_q_hahn import DualQHahn
from .dual_q_krawtchouk import DualQKrawtchouk
from .little_q_jacobi import LittleQJacobi
from .q_charlier import QCharlier
from .q_hahn import QHahn
from .q_krawtchouk import QKrawtchouk
from .q_meixner import QMeixner
from .q_racah import QRacah
from .quantum_q_krawtchouk import QuantumQKrawtchouk


__all__ = [
    # Base class
    "QPolynomialBase",
    # Askey-Wilson class
    "QRacah",
    "QHahn",
    "DualQHahn",
    # q-Hahn class
    "BigQJacobi",
    "LittleQJacobi",
    "QMeixner",
    "QKrawtchouk",
    "QCharlier",
    "DualQKrawtchouk",
    "AffineQKrawtchouk",
    "QuantumQKrawtchouk",
    # Hermite variants
    "DiscreteQHermite1",
    "DiscreteQHermite2",
    "ContinuousQHermite",
    # Continuous q-polynomials
    "ContinuousQJacobi",
    "ContinuousQUltraspherical",
    "ContinuousQLaguerre",
    "ContinuousQLegendre",
]
