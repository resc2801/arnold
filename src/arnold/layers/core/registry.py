# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Layer registry for ARNOLD KAN layers.

This module provides a centralized registry mapping string names to layer classes,
enabling dynamic layer instantiation via string identifiers. This is useful for:

- Configuration-driven model building
- Hyperparameter search over layer types
- Serialization/deserialization of layer configurations
- User-friendly API with intuitive naming

Usage
-----
>>> from arnold.layers.core.registry import get_layer, list_layers
>>> layer = get_layer("legendre", degree=5, units=32)
>>> print(list_layers())

The registry supports multiple aliases for the same layer class:

>>> get_layer("chebyshev", ...)      # Returns Chebyshev1st
>>> get_layer("chebyshev_t", ...)    # Returns Chebyshev1st (alias)
>>> get_layer("ultraspherical", ...) # Returns Gegenbauer (alias)
"""

from typing import Any

from .kan_base import KANBase

# =============================================================================
# Polynomial Layers
# =============================================================================

# Jacobi Family (orthogonal/)
from .polynomial.orthogonal import (
    Jacobi,
    Legendre,
    Gegenbauer,
    Chebyshev1st,
    Chebyshev2nd,
    Chebyshev3rd,
    Chebyshev4th,
)

# Laguerre Family (orthogonal/)
from .polynomial.orthogonal import GeneralizedLaguerre

# Hermite Family (orthogonal/)
from .polynomial.orthogonal import Hermite

# Bessel Family (orthogonal/)
from .polynomial.orthogonal import Bessel

# Meixner-Pollaczek Family (orthogonal/)
from .polynomial.orthogonal import AssociatedMeixnerPollaczek

# Pollaczek (orthogonal/)
from .polynomial.orthogonal import Pollaczek

# Wilson (orthogonal/)
from .polynomial.orthogonal import Wilson

# Al-Salam-Carlitz (orthogonal/)
from .polynomial.orthogonal import AlSalamCarlitz1st, AlSalamCarlitz2nd

# Askey-Wilson (orthogonal/)
from .polynomial.orthogonal import AskeyWilson

# Bannai-Ito (orthogonal/)
from .polynomial.orthogonal import BannaiIto

# Charlier (orthogonal/)
from .polynomial.orthogonal import Charlier

# q-Orthogonal Polynomials (q_orthogonal/)
from .polynomial.q_orthogonal import (
    QPolynomialBase,
    QHahn,
    BigQJacobi,
    LittleQJacobi,
    QMeixner,
    QKrawtchouk,
    QCharlier,
    QRacah,
    DualQHahn,
    DualQKrawtchouk,
    AffineQKrawtchouk,
    DiscreteQHermite1,
    DiscreteQHermite2,
    ContinuousQHermite,
    ContinuousQJacobi,
    ContinuousQUltraspherical,
    QuantumQKrawtchouk,
    ContinuousQLaguerre,
    ContinuousQLegendre,
)

# Discrete Orthogonal (discrete.py)
from .polynomial.discrete import Hahn, Krawtchouk, Meixner, Racah

# Continuous Hahn Family (continuous_hahn.py)
from .polynomial.continuous_hahn import (
    ContinuousHahn,
    ContinuousDualHahn,
    DualHahn,
    StieltjesWigert,
)

# N-bonacci Sequences (sequences/)
from .polynomial.sequences import (
    Fibonacci,
    Tribonacci,
    Tetranacci,
    Pentanacci,
    Hexanacci,
    Heptanacci,
    Octanacci,
)

# W-Polynomials / Lucas-type (sequences/)
from .polynomial.sequences import (
    Lucas,
    Pell,
    PellLucas,
    Fermat,
    FermatLucas,
    Jacobsthal,
    JacobsthalLucas,
)

# Non-Orthogonal (non_orthogonal.py)
from .polynomial.non_orthogonal import Boubaker

# Zernike (zernike.py)
from .polynomial.zernike import Zernike

# Rational Functions
from .rational_functions import Laurent

# =============================================================================
# Radial Basis Functions
# =============================================================================

from .rbf import (
    RBFBase,
    GaussianRBF,
    MultiquadricRBF,
    InverseMultiQuadricRBF,
    ThinPlateSplineRBF,
    CauchyRBF,
    LinearRBF,
    CubicRBF,
    InverseQuadricRBF,
    ExponentialRBF,
    PowerRBF,
)

# =============================================================================
# Splines
# =============================================================================

from .splines import SplineBase, BSpline, CatmullRom, Cardinal

# =============================================================================
# Wavelets
# =============================================================================

from .wavelets import (
    WaveletBase,
    Haar,
    Daubechies,
    Symlet,
    Coiflet,
    Ricker,
    Morelet,
    Shannon,
    Meyer,
    Bump,
    Poisson,
    DerivativeOfGaussian,
)

# =============================================================================
# Spectral Layers
# =============================================================================

from .spectral import (
    SpectralBase,
    FourierKAN,
    RandomFourierFeatures,
)


# =============================================================================
# Layer Registry
# =============================================================================

LAYER_REGISTRY: dict[str, type[KANBase]] = {
    # -------------------------------------------------------------------------
    # Jacobi Family (Classical Orthogonal)
    # -------------------------------------------------------------------------
    "jacobi": Jacobi,
    "legendre": Legendre,
    "legendre_kan": Legendre,  # Alias
    "gegenbauer": Gegenbauer,
    "ultraspherical": Gegenbauer,  # Mathematical alias
    # Chebyshev polynomials
    "chebyshev": Chebyshev1st,  # Default: T_n
    "chebyshev1": Chebyshev1st,
    "chebyshev_t": Chebyshev1st,  # T_n notation
    "chebyshev1st": Chebyshev1st,
    "chebyshev2": Chebyshev2nd,
    "chebyshev_u": Chebyshev2nd,  # U_n notation
    "chebyshev2nd": Chebyshev2nd,
    "chebyshev3": Chebyshev3rd,
    "chebyshev_v": Chebyshev3rd,  # V_n notation
    "chebyshev3rd": Chebyshev3rd,
    "chebyshev4": Chebyshev4th,
    "chebyshev_w": Chebyshev4th,  # W_n notation
    "chebyshev4th": Chebyshev4th,
    # -------------------------------------------------------------------------
    # Laguerre, Hermite, Bessel
    # -------------------------------------------------------------------------
    "laguerre": GeneralizedLaguerre,
    "generalized_laguerre": GeneralizedLaguerre,
    "hermite": Hermite,
    "bessel": Bessel,
    # -------------------------------------------------------------------------
    # Other Classical Orthogonal
    # -------------------------------------------------------------------------
    "meixner_pollaczek": AssociatedMeixnerPollaczek,
    "associated_meixner_pollaczek": AssociatedMeixnerPollaczek,
    "pollaczek": Pollaczek,
    "wilson": Wilson,
    "al_salam_carlitz_1": AlSalamCarlitz1st,
    "al_salam_carlitz_2": AlSalamCarlitz2nd,
    "askey_wilson": AskeyWilson,
    "bannai_ito": BannaiIto,
    "charlier": Charlier,
    # -------------------------------------------------------------------------
    # Discrete Orthogonal
    # -------------------------------------------------------------------------
    "hahn": Hahn,
    "krawtchouk": Krawtchouk,
    "kravchuk": Krawtchouk,  # Alternative spelling
    "meixner": Meixner,
    "racah": Racah,
    # -------------------------------------------------------------------------
    # Continuous Hahn Family
    # -------------------------------------------------------------------------
    "continuous_hahn": ContinuousHahn,
    "continuous_dual_hahn": ContinuousDualHahn,
    "dual_hahn": DualHahn,
    "stieltjes_wigert": StieltjesWigert,
    # -------------------------------------------------------------------------
    # q-Orthogonal Polynomials
    # -------------------------------------------------------------------------
    "q_hahn": QHahn,
    "big_q_jacobi": BigQJacobi,
    "little_q_jacobi": LittleQJacobi,
    "q_meixner": QMeixner,
    "q_krawtchouk": QKrawtchouk,
    "q_charlier": QCharlier,
    "q_racah": QRacah,
    "dual_q_hahn": DualQHahn,
    "dual_q_krawtchouk": DualQKrawtchouk,
    "affine_q_krawtchouk": AffineQKrawtchouk,
    "discrete_q_hermite_1": DiscreteQHermite1,
    "discrete_q_hermite_2": DiscreteQHermite2,
    "continuous_q_hermite": ContinuousQHermite,
    "continuous_q_jacobi": ContinuousQJacobi,
    "continuous_q_ultraspherical": ContinuousQUltraspherical,
    "rogers": ContinuousQUltraspherical,  # Mathematical alias
    "quantum_q_krawtchouk": QuantumQKrawtchouk,
    "continuous_q_laguerre": ContinuousQLaguerre,
    "continuous_q_legendre": ContinuousQLegendre,
    # -------------------------------------------------------------------------
    # N-bonacci Sequences
    # -------------------------------------------------------------------------
    "fibonacci": Fibonacci,
    "fib": Fibonacci,  # Short alias
    "tribonacci": Tribonacci,
    "tetranacci": Tetranacci,
    "pentanacci": Pentanacci,
    "hexanacci": Hexanacci,
    "heptanacci": Heptanacci,
    "octanacci": Octanacci,
    # -------------------------------------------------------------------------
    # W-Polynomials (Lucas-type)
    # -------------------------------------------------------------------------
    "lucas": Lucas,
    "pell": Pell,
    "pell_lucas": PellLucas,
    "fermat": Fermat,
    "fermat_lucas": FermatLucas,
    "jacobsthal": Jacobsthal,
    "jacobsthal_lucas": JacobsthalLucas,
    # -------------------------------------------------------------------------
    # Non-Orthogonal & Special
    # -------------------------------------------------------------------------
    "boubaker": Boubaker,
    "zernike": Zernike,
    "laurent": Laurent,
    # -------------------------------------------------------------------------
    # Radial Basis Functions
    # -------------------------------------------------------------------------
    "gaussian_rbf": GaussianRBF,
    "gaussian": GaussianRBF,  # Short alias
    "rbf": GaussianRBF,  # Default RBF
    "multiquadric": MultiquadricRBF,
    "multiquadric_rbf": MultiquadricRBF,
    "inverse_multiquadric": InverseMultiQuadricRBF,
    "inverse_multiquadric_rbf": InverseMultiQuadricRBF,
    "imq": InverseMultiQuadricRBF,  # Short alias
    "thin_plate_spline": ThinPlateSplineRBF,
    "thin_plate_spline_rbf": ThinPlateSplineRBF,
    "tps": ThinPlateSplineRBF,  # Short alias
    "cauchy_rbf": CauchyRBF,
    "cauchy": CauchyRBF,
    "linear_rbf": LinearRBF,
    "cubic_rbf": CubicRBF,
    "inverse_quadric": InverseQuadricRBF,
    "inverse_quadric_rbf": InverseQuadricRBF,
    "exponential_rbf": ExponentialRBF,
    "exponential": ExponentialRBF,
    "power_rbf": PowerRBF,
    # -------------------------------------------------------------------------
    # Splines
    # -------------------------------------------------------------------------
    "bspline": BSpline,
    "b_spline": BSpline,
    "catmull_rom": CatmullRom,
    "catmullrom": CatmullRom,
    "cardinal": Cardinal,
    "cardinal_spline": Cardinal,
    # -------------------------------------------------------------------------
    # Wavelets
    # -------------------------------------------------------------------------
    "haar": Haar,
    "haar_wavelet": Haar,
    "daubechies": Daubechies,
    "db": Daubechies,  # Short alias
    "symlet": Symlet,
    "sym": Symlet,  # Short alias
    "coiflet": Coiflet,
    "coif": Coiflet,  # Short alias
    "ricker": Ricker,
    "mexican_hat": Ricker,  # Common alias
    "morlet": Morelet,
    "morelet": Morelet,
    "shannon": Shannon,
    "shannon_wavelet": Shannon,
    "meyer": Meyer,
    "meyer_wavelet": Meyer,
    "bump": Bump,
    "bump_wavelet": Bump,
    "poisson": Poisson,
    "poisson_wavelet": Poisson,
    "dog": DerivativeOfGaussian,
    "derivative_of_gaussian": DerivativeOfGaussian,
    # -------------------------------------------------------------------------
    # Spectral
    # -------------------------------------------------------------------------
    "fourier": FourierKAN,
    "fourier_kan": FourierKAN,
    "trigonometric": FourierKAN,  # Mathematical alias
    "rff": RandomFourierFeatures,
    "random_fourier_features": RandomFourierFeatures,
    "random_fourier": RandomFourierFeatures,
}


# =============================================================================
# Category Mapping
# =============================================================================

LAYER_CATEGORIES: dict[str, list[str]] = {
    "polynomial": [
        "jacobi", "legendre", "gegenbauer", "chebyshev1", "chebyshev2",
        "chebyshev3", "chebyshev4", "laguerre", "hermite", "bessel",
        "meixner_pollaczek", "pollaczek", "wilson", "al_salam_carlitz_1",
        "al_salam_carlitz_2", "askey_wilson", "bannai_ito", "charlier",
        "boubaker", "zernike", "laurent",
    ],
    "discrete_orthogonal": [
        "hahn", "krawtchouk", "meixner", "racah",
        "continuous_hahn", "continuous_dual_hahn", "dual_hahn", "stieltjes_wigert",
    ],
    "q_orthogonal": [
        "q_hahn", "big_q_jacobi", "little_q_jacobi", "q_meixner", "q_krawtchouk",
        "q_charlier", "q_racah", "dual_q_hahn", "dual_q_krawtchouk",
        "affine_q_krawtchouk", "discrete_q_hermite_1", "discrete_q_hermite_2",
        "continuous_q_hermite", "continuous_q_jacobi", "continuous_q_ultraspherical",
        "quantum_q_krawtchouk", "continuous_q_laguerre", "continuous_q_legendre",
    ],
    "fibonacci_type": [
        "fibonacci", "tribonacci", "tetranacci", "pentanacci",
        "hexanacci", "heptanacci", "octanacci",
    ],
    "lucas_type": [
        "lucas", "pell", "pell_lucas", "fermat", "fermat_lucas",
        "jacobsthal", "jacobsthal_lucas",
    ],
    "rbf": [
        "gaussian_rbf", "multiquadric", "inverse_multiquadric",
        "thin_plate_spline", "cauchy_rbf", "linear_rbf", "cubic_rbf",
        "inverse_quadric", "exponential_rbf", "power_rbf",
    ],
    "spline": [
        "bspline", "catmull_rom", "cardinal",
    ],
    "wavelet": [
        "haar", "daubechies", "symlet", "coiflet", "ricker",
        "morlet", "shannon", "meyer", "bump", "poisson", "dog",
    ],
    "spectral": [
        "fourier", "rff",
    ],
}


# =============================================================================
# Public API
# =============================================================================

def get_layer(name: str, **kwargs: Any) -> KANBase:
    r"""
    Instantiate a KAN layer by name.

    Parameters
    ----------
    name : str
        The name of the layer (case-insensitive). Use :func:`list_layers`
        to see all available names.
    **kwargs
        Keyword arguments passed to the layer constructor.

    Returns
    -------
    KANBase
        An instance of the requested layer.

    Raises
    ------
    ValueError
        If the layer name is not recognized.

    Examples
    --------
    >>> layer = get_layer("legendre", degree=5, units=32)
    >>> layer = get_layer("chebyshev", degree=10, units=64)
    >>> layer = get_layer("gaussian_rbf", units=128, num_centers=50)

    See Also
    --------
    list_layers : List all available layer names.
    list_layers_by_category : List layers grouped by category.
    """
    name_lower = name.lower().replace("-", "_").replace(" ", "_")

    if name_lower not in LAYER_REGISTRY:
        available = ", ".join(sorted(set(LAYER_REGISTRY.keys())))
        raise ValueError(
            f"Unknown layer: '{name}'. "
            f"Available layers: {available}"
        )

    layer_cls = LAYER_REGISTRY[name_lower]
    return layer_cls(**kwargs)


def list_layers(include_aliases: bool = False) -> list[str]:
    r"""
    List all available layer names.

    Parameters
    ----------
    include_aliases : bool, default=False
        If True, include all aliases. If False, return only canonical names
        (one per layer class).

    Returns
    -------
    list[str]
        Sorted list of layer names.

    Examples
    --------
    >>> names = list_layers()
    >>> print(len(names))  # Number of unique layer classes
    >>> names_with_aliases = list_layers(include_aliases=True)
    """
    if include_aliases:
        return sorted(LAYER_REGISTRY.keys())

    # Return only canonical names (first occurrence of each class)
    seen_classes: set[type] = set()
    canonical: list[str] = []

    for name, cls in sorted(LAYER_REGISTRY.items()):
        if cls not in seen_classes:
            canonical.append(name)
            seen_classes.add(cls)

    return sorted(canonical)


def list_layers_by_category() -> dict[str, list[str]]:
    r"""
    List layers grouped by category.

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping category names to lists of layer names.

    Examples
    --------
    >>> categories = list_layers_by_category()
    >>> print(categories["polynomial"])
    >>> print(categories["rbf"])
    """
    return LAYER_CATEGORIES.copy()


def get_layer_class(name: str) -> type[KANBase]:
    r"""
    Get the layer class (not an instance) by name.

    Parameters
    ----------
    name : str
        The name of the layer (case-insensitive).

    Returns
    -------
    type[KANBase]
        The layer class.

    Raises
    ------
    ValueError
        If the layer name is not recognized.

    Examples
    --------
    >>> LegendreClass = get_layer_class("legendre")
    >>> layer = LegendreClass(degree=5, units=32)
    """
    name_lower = name.lower().replace("-", "_").replace(" ", "_")

    if name_lower not in LAYER_REGISTRY:
        available = ", ".join(sorted(set(LAYER_REGISTRY.keys())))
        raise ValueError(
            f"Unknown layer: '{name}'. "
            f"Available layers: {available}"
        )

    return LAYER_REGISTRY[name_lower]


def is_registered(name: str) -> bool:
    r"""
    Check if a layer name is registered.

    Parameters
    ----------
    name : str
        The name to check (case-insensitive).

    Returns
    -------
    bool
        True if the name is registered, False otherwise.

    Examples
    --------
    >>> is_registered("legendre")
    True
    >>> is_registered("unknown_layer")
    False
    """
    name_lower = name.lower().replace("-", "_").replace(" ", "_")
    return name_lower in LAYER_REGISTRY


def get_aliases(layer_class: type[KANBase]) -> list[str]:
    r"""
    Get all registered aliases for a layer class.

    Parameters
    ----------
    layer_class : type[KANBase]
        The layer class.

    Returns
    -------
    list[str]
        Sorted list of all names that map to this class.

    Examples
    --------
    >>> from arnold.layers.core.polynomial.orthogonal import Gegenbauer
    >>> aliases = get_aliases(Gegenbauer)
    >>> print(aliases)  # ['gegenbauer', 'ultraspherical']
    """
    return sorted(
        name for name, cls in LAYER_REGISTRY.items()
        if cls is layer_class
    )


__all__ = [
    "LAYER_REGISTRY",
    "LAYER_CATEGORIES",
    "get_layer",
    "get_layer_class",
    "list_layers",
    "list_layers_by_category",
    "is_registered",
    "get_aliases",
]
