## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.

"""
Polynomial KAN layers and bases.
"""

from ..rational_functions import Laurent
from .continuous_hahn import (
    ContinuousDualHahn,
    ContinuousHahn,
    DualHahn,
    StieltjesWigert,
)
from .discrete import (
    Hahn,
    Krawtchouk,
    Meixner,
    Racah,
)
from .n_bonacci import (
    Fibonacci,
    Heptanacci,
    Hexanacci,
    Octanacci,
    Pentanacci,
    Tetranacci,
    Tribonacci,
)
from .non_orthogonal import Boubaker
from .orthogonal import (
    AlSalamCarlitz1st,
    AlSalamCarlitz2nd,
    AskeyWilson,
    AssociatedMeixnerPollaczek,
    BannaiIto,
    Bessel,
    Charlier,
    Chebyshev1st,
    Chebyshev2nd,
    Chebyshev3rd,
    Chebyshev4th,
    Gegenbauer,
    GeneralizedLaguerre,
    Hermite,
    Jacobi,
    Legendre,
    Pollaczek,
    Wilson,
)
from .poly_base import PolynomialBase
from .q_polynomials import (
    AffineQKrawtchouk,
    BigQJacobi,
    ContinuousQHermite,
    ContinuousQJacobi,
    ContinuousQLaguerre,
    ContinuousQLegendre,
    ContinuousQUltraspherical,
    DiscreteQHermite1,
    DiscreteQHermite2,
    DualQHahn,
    DualQKrawtchouk,
    LittleQJacobi,
    QCharlier,
    QHahn,
    QKrawtchouk,
    QMeixner,
    QPolynomialBase,
    QRacah,
    QuantumQKrawtchouk,
)
from .w_polynomials import (
    Fermat,
    FermatLucas,
    Jacobsthal,
    JacobsthalLucas,
    Lucas,
    Pell,
    PellLucas,
)
from .zernike import Zernike


__all__ = [
    "PolynomialBase",
    "Boubaker",
    "Laurent",
    "Lucas",
    "Fermat",
    "FermatLucas",
    "Jacobsthal",
    "JacobsthalLucas",
    "Pell",
    "PellLucas",
    "Fibonacci",
    "Heptanacci",
    "Hexanacci",
    "Octanacci",
    "Pentanacci",
    "Tetranacci",
    "AlSalamCarlitz1st",
    "AlSalamCarlitz2nd",
    "AskeyWilson",
    "AssociatedMeixnerPollaczek",
    "BannaiIto",
    "Bessel",
    "Charlier",
    "Chebyshev1st",
    "Chebyshev2nd",
    "Chebyshev3rd",
    "Chebyshev4th",
    "Gegenbauer",
    "GeneralizedLaguerre",
    "Hermite",
    "Jacobi",
    "Legendre",
    "Pollaczek",
    "Wilson",
    # Discrete orthogonal polynomials
    "Hahn",
    "Krawtchouk",
    "Meixner",
    "Racah",
    # Continuous Hahn family (Wilson class)
    "ContinuousHahn",
    "ContinuousDualHahn",
    "DualHahn",
    "StieltjesWigert",
    # q-Hahn class (q-orthogonal polynomials) - Sprint 7E
    "QPolynomialBase",
    "QHahn",
    "BigQJacobi",
    "LittleQJacobi",
    "QMeixner",
    "QKrawtchouk",
    # q-Polynomials Part 2 - Sprint 7F
    "QCharlier",
    "QRacah",
    "DualQHahn",
    "DualQKrawtchouk",
    "AffineQKrawtchouk",
    # q-Polynomials Part 3 (Askey-Wilson Class) - Sprint 7G
    "DiscreteQHermite1",
    "DiscreteQHermite2",
    "ContinuousQHermite",
    "ContinuousQJacobi",
    "ContinuousQUltraspherical",
    # q-Polynomials Part 4 & Special - Sprint 7H
    "QuantumQKrawtchouk",
    "ContinuousQLaguerre",
    "ContinuousQLegendre",
    "Tribonacci",
    "Zernike",
]
