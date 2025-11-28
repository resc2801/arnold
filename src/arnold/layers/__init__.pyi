# Copyright (c) 2025 René Schubotz. All rights reserved.
# Type stubs for arnold.layers

from arnold.layers import convolutional as convolutional
from arnold.layers import core as core

# Re-export all core layer classes
from arnold.layers.core import (
    KANBase as KANBase,
    PolynomialBase as PolynomialBase,
    RBFBase as RBFBase,
    WaveletBase as WaveletBase,
    detect_hardware as detect_hardware,
    get_recommended_dtype as get_recommended_dtype,
    # Wavelets
    Bump as Bump,
    DerivativeOfGaussian as DerivativeOfGaussian,
    Meyer as Meyer,
    Morelet as Morelet,
    Poisson as Poisson,
    Ricker as Ricker,
    Shannon as Shannon,
    # Splines
    BSpline as BSpline,
    CatmullRom as CatmullRom,
    Cardinal as Cardinal,
    # Fibonacci-type polynomials
    Fibonacci as Fibonacci,
    Heptanacci as Heptanacci,
    Hexanacci as Hexanacci,
    Octanacci as Octanacci,
    Pentanacci as Pentanacci,
    Tetranacci as Tetranacci,
    Lucas as Lucas,
    Fermat as Fermat,
    FermatLucas as FermatLucas,
    Jacobsthal as Jacobsthal,
    JacobsthalLucas as JacobsthalLucas,
    Pell as Pell,
    PellLucas as PellLucas,
    # Orthogonal polynomials
    AlSalamCarlitz1st as AlSalamCarlitz1st,
    AlSalamCarlitz2nd as AlSalamCarlitz2nd,
    AskeyWilson as AskeyWilson,
    AssociatedMeixnerPollaczek as AssociatedMeixnerPollaczek,
    BannaiIto as BannaiIto,
    Bessel as Bessel,
    Charlier as Charlier,
    Chebyshev1st as Chebyshev1st,
    Chebyshev2nd as Chebyshev2nd,
    Chebyshev3rd as Chebyshev3rd,
    Chebyshev4th as Chebyshev4th,
    Gegenbauer as Gegenbauer,
    GeneralizedLaguerre as GeneralizedLaguerre,
    Hermite as Hermite,
    Jacobi as Jacobi,
    Legendre as Legendre,
    Pollaczek as Pollaczek,
    Wilson as Wilson,
    # Discrete orthogonal polynomials
    Krawtchouk as Krawtchouk,
    Hahn as Hahn,
    Meixner as Meixner,
    Racah as Racah,
    # Other polynomials
    Boubaker as Boubaker,
    Laurent as Laurent,
    # RBFs
    ExponentialRBF as ExponentialRBF,
    CauchyRBF as CauchyRBF,
    CubicRBF as CubicRBF,
    GaussianRBF as GaussianRBF,
    InverseMultiQuadricRBF as InverseMultiQuadricRBF,
    InverseQuadricRBF as InverseQuadricRBF,
    LinearRBF as LinearRBF,
    MultiquadricRBF as MultiquadricRBF,
    PowerRBF as PowerRBF,
    ThinPlateSplineRBF as ThinPlateSplineRBF,
)

__all__: list[str]
