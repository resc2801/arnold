"""
Implementation of KAN layers.
"""

from .polynomial.n_bonacci import (
    Fibonacci,
    Heptanacci,
    Hexanacci,
    Octanacci,
    Pentanacci,
    Tetranacci,
)

from .polynomial.w_polynomials import (
    Lucas,
    Fermat,
    FermatLucas,
    Jacobsthal,
    JacobsthalLucas,
    Pell,
    PellLucas
)

from .polynomial.orthogonal import (
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

from .polynomial.non_orthogonal import (
    Boubaker
)

from .rational_functions import (
    Laurent
)

from .radial_basis_functions import (
    ExponentialRBF, 
    CauchyRBF, 
    CubicRBF, 
    GaussianRBF, 
    InverseMultiQuadricRBF, 
    InverseQuadricRBF, 
    LinearRBF, 
    MultiquadricRBF, 
    PowerRBF, 
    ThinPlateSplineRBF,
)

from .wavelets import (
    Bump,
    DerivativeOfGaussian,
    Meyer,
    Morelet,
    Poisson,
    Ricker,
    Shannon,
)

__all__ = [
    # Wavelet KAN layers
    "Bump",
    "DerivativeOfGaussian",
    "Meyer",
    "Morelet",
    "Poisson",
    "Ricker",
    "Shannon",

    # N-bonacci polynomial KAN layers
    "Fibonacci",
    "Heptanacci",
    "Hexanacci",
    "Octanacci",
    "Pentanacci",
    "Tetranacci",

    # W polynomial KAN layers
    "Lucas",
    "Fermat",
    "FermatLucas",
    "Jacobsthal",
    "JacobsthalLucas",
    "Pell",
    "PellLucas",

    # Orthogonal polynomial KAN layers
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

    # Nonorthogonal KAN layers
    "Boubaker",
    
    # Rational function KAN layers
    "Laurent",

    # Radial Basis Function KAN layers
    "ExponentialRBF", 
    "CauchyRBF", 
    "CubicRBF", 
    "GaussianRBF", 
    "InverseMultiQuadricRBF", 
    "InverseQuadricRBF", 
    "LinearRBF", 
    "MultiquadricRBF", 
    "PowerRBF", 
    "ThinPlateSplineRBF",
]