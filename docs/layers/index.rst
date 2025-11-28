.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _layers-index:

====================================================
Part III: Layer Reference
====================================================

This part provides comprehensive API documentation for all KAN layers in ARNOLD.
Each layer family is documented with mathematical foundations, parameters, and
usage examples.

.. toctree::
   :maxdepth: 2
   :caption: Layer Families

   polynomial_kan_layers
   radial_basis_kan_layers
   wavelet_kan_layers
   spline_kan_layers
   spectral_kan_layers
   geometric_kan_layers
   special_kan_layers
   symbolic_tools

----

Layer Overview
--------------

ARNOLD provides **80+ KAN layer implementations** organized into multiple families:

**Polynomial KAN Layers** (:doc:`polynomial_kan_layers`)

Polynomial bases with strong theoretical foundations. Includes:

- Classical orthogonal: Chebyshev, Legendre, Hermite, Laguerre, Jacobi
- Discrete orthogonal: Charlier, Meixner, Krawtchouk
- q-Polynomials: Al-Salam-Carlitz, Askey-Wilson
- Lucas sequences: Fibonacci, Pell, Fermat
- Non-orthogonal: Boubaker, Laurent

**Radial Basis Function KAN Layers** (:doc:`radial_basis_kan_layers`)

Localized basis functions for meshfree approximation:

- Gaussian RBF
- Multiquadric / Inverse Multiquadric
- Thin-Plate Spline
- Cauchy, Power, Cubic, Linear

**Wavelet KAN Layers** (:doc:`wavelet_kan_layers`)

Multi-scale basis functions for time-frequency analysis:

- Ricker (Mexican Hat)
- Morlet
- Meyer, Shannon
- Bump, Derivative of Gaussian
- Haar, Daubechies, Symlet, Coiflet

**Spline KAN Layers** (:doc:`spline_kan_layers`)

Smooth piecewise-polynomial basis functions:

- B-Spline (Cox-de Boor)
- Catmull-Rom (interpolating)
- Cardinal (adjustable tension)

**Spectral KAN Layers** (:doc:`spectral_kan_layers`)

Trigonometric and random Fourier basis functions:

- FourierKAN (sine/cosine series)
- RandomFourierFeatures (kernel approximation)

**Geometric KAN Layers** (:doc:`geometric_kan_layers`)

Basis functions for spherical and geometric domains:

- Zernike (unit disk, optics)
- SphericalHarmonics (S²)
- HypersphericalHarmonics (Sⁿ⁻¹)

**Special Function KAN Layers** (:doc:`special_kan_layers`)

Classical special functions as basis:

- Airy (quantum mechanics, optics)
- Bessel (cylindrical symmetry)
- And more (Mathieu, Whittaker, etc.)

**Symbolic Tools** (:doc:`symbolic_tools`)

Convert trained KANs to mathematical expressions:

- kan_to_polynomial() — Extract symbolic formulas
- kan_to_latex() — Export for publications
- Simplification utilities

----

Common Layer API
----------------

All KAN layers share a common API inspired by Keras conventions:

**Constructor Parameters:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``units``
     - int
     - Output dimension (like ``Dense``)
   * - ``use_bias``
     - bool
     - Whether to add a bias term (default: True)
   * - ``kernel_initializer``
     - str/Initializer
     - Initializer for coefficients
   * - ``kernel_regularizer``
     - Regularizer
     - Regularization for coefficients
   * - ``input_clip``
     - tuple/None
     - Clip inputs to (min, max) for stability

**Polynomial-Specific:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``degree``
     - int
     - Maximum polynomial degree
   * - ``core_ranks``
     - tuple/None
     - Tucker decomposition ranks for memory efficiency
   * - ``use_clenshaw``
     - bool/None
     - Force or disable Clenshaw recurrence

**RBF-Specific:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``num_grids``
     - int
     - Number of RBF centers
   * - ``grid_min``
     - float
     - Grid lower bound
   * - ``grid_max``
     - float
     - Grid upper bound

**Wavelet-Specific:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``sigma_init``
     - float
     - Initial scale parameter
   * - ``omega0``
     - float
     - Central frequency (Morlet)

----

Quick Import Guide
------------------

**Import individual layers:**

.. code-block:: python

   from arnold.layers import Legendre, Chebyshev1st, GaussianRBF, Ricker

**Import entire family:**

.. code-block:: python

   from arnold.layers.core.polynomial import orthogonal, non_orthogonal, n_bonacci
   from arnold.layers.core.radial_basis_functions import *
   from arnold.layers.core.wavelets import *

**Check available layers:**

.. code-block:: python

   import arnold
   print(arnold.layers.__all__)  # List all public layers

----

Layer Selection Flowchart
-------------------------

.. code-block:: text

   Is your function smooth?
   │
   ├─► Yes: Is data bounded in [-1, 1]?
   │       │
   │       ├─► Yes: Use Chebyshev1st or Legendre
   │       │
   │       └─► No: Is it Gaussian-weighted?
   │               │
   │               ├─► Yes: Use Hermite
   │               │
   │               └─► No: Use Laguerre or normalize first
   │
   └─► No: Is it multi-scale or time-varying?
           │
           ├─► Yes: Use Wavelet (Ricker, Morlet)
           │
           └─► No: Does it have local features/outliers?
                   │
                   ├─► Yes: Use RBF (Gaussian, Cauchy)
                   │
                   └─► No: Consider hybrid approach

See :doc:`../basis_selection_guide` for detailed guidance.

----

Performance Characteristics
---------------------------

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Layer Family
     - Param Count
     - GPU Friendly
     - TPU Friendly
     - Mixed Precision
   * - Polynomial
     - O(n·m·d)
     - ✓✓✓
     - ✓✓
     - ✓ (use Clenshaw)
   * - RBF
     - O(n·m·K)
     - ✓✓✓
     - ✓✓✓
     - ✓✓✓
   * - Wavelet
     - O(n·m·J·K)
     - ✓✓✓
     - ✓✓✓
     - ✓✓✓

Where:
- n = input dimension
- m = output dimension (units)
- d = polynomial degree
- K = number of RBF grids
- J, K = wavelet scales and translations

With Tucker decomposition, polynomial parameter count becomes O(r₁r₂r₃ + nr₁ + mr₂ + dr₃).
