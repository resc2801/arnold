# Changelog

All notable changes to ARNOLD will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### New Layer Packages
- **Spectral layers** (`arnold.layers.spectral`):
  - `FourierKAN`: Trigonometric basis {1, cos, sin} with learnable frequency
  - `RandomFourierFeatures`: RBF kernel approximation via random Fourier features

- **Geometric layers** (`arnold.layers.geometric`):
  - `Zernike`: Zernike radial polynomials for optical/disk domains
  - `SphericalHarmonics`: Legendre-based harmonics on S²
  - `HypersphericalHarmonics`: Gegenbauer-based harmonics for n-dimensional spheres

- **Special function layers** (`arnold.layers.special`):
  - `Airy`: Airy functions Ai(x), Bi(x) via power series
  - `BesselFunctions`: Bessel functions J_ν(x) via series expansion
  - Stubs for: ParabolicCylinder, Mathieu, Whittaker, Slepian, LegendreFunctions, EllipticFunctions

- **Symbolic tools** (`arnold.layers.symbolic`):
  - `kan_to_polynomial()`: Convert trained KAN to SymPy expression ⭐
  - `kan_to_latex()`: Export to LaTeX for publications
  - `extract_coefficients()`: Get raw weight values
  - Simplification utilities: simplify, expand, factor, collect

- **Training utilities**:
  - `arnold.layers.constraints`: KANConstraint, Bounds, Positivity, Monotonicity, Orthogonality
  - `arnold.layers.regularizers`: KANRegularizer, L1/L2/L1L2, Sparsity, Smoothness, Curvature
  - `arnold.layers.initializers`: KANInitializer, Polynomial, RBF, Spectral, OrthogonalInit

- **Advanced architectures** (`arnold.layers.architectures`):
  - `OriginalKAN`: B-spline KAN from Liu et al. (2024)
  - `CompactKAN`: Efficient KAN with configurable basis + residual
  - `KalmanKAN`: Recursive filter-based KAN for sequences
  - `MLPBasis`: MLP as learnable basis function
  - `HyperKAN`: Hypernetwork-based dynamic weight generation

- **Mixed basis layers** (`arnold.layers.mixed`):
  - `MixedBasis`: Combine multiple basis families with learned softmax weights
  - `ProductBasis`: Tensor product expansions (Legendre × Fourier)
  - `AttentionBasis`: Attention-weighted dynamic basis combination

#### Registry & API
- Centralized layer registry with 130+ string→class mappings
- Factory functions: `get_layer()`, `list_layers()`, `get_layer_class()`
- Case-insensitive lookup with alias support
- 9 categories: polynomial, q_orthogonal, rbf, wavelet, spline, spectral, geometric, special, rational

### Changed

#### Architecture Refactoring (MASTERPLAN Migration)
- Split monolithic modules into focused, single-responsibility files
- Established clean import DAG: `symbolic/ → core/* → common/`
- Created shared utilities in `common/` (types, evaluation, parameters, numerics)
- Polynomial layers reorganized into `orthogonal/`, `q_orthogonal/`, `sequences/`
- RBF layers split into 12 focused modules
- Wavelet layers split into 14 focused modules (+ coefficients.py)
- Spline layers split into 5 focused modules

#### Code Quality
- All modules follow strict import DAG (no cycles)
- Comprehensive Sphinx docstrings with math notation
- Consistent copyright headers on all source files
- 1428 tests passing (up from ~1200)

### Documentation
- New RST docs for spectral, geometric, special, and symbolic layers
- Updated layer index: 80+ layers documented across 8 families
- README.md updated with new features
- Docs build successfully with Sphinx

### Fixed
- Zernike R_n^0 formula corrected to use P_n(2ρ²-1)
- Import cycle prevention throughout codebase

## [0.1.0] - 2025-XX-XX

### Added
- First public release
- 80+ KAN layer implementations
- Full Keras API compliance
- Documentation and examples
- Symbolic expression extraction (kan_to_polynomial)

[Unreleased]: https://github.com/resc2801/arnold/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/resc2801/arnold/releases/tag/v0.1.0
