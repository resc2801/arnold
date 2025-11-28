.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _basis-selection-guide:

================================================
Basis Selection Guide
================================================

This chapter helps you choose the right basis function for your KAN layer.
The choice of basis fundamentally affects approximation quality, numerical 
stability, and computational efficiency.

.. contents:: In This Chapter
   :local:
   :depth: 2

----

Quick Reference
---------------

.. list-table:: Basis Selection at a Glance
   :widths: 25 25 25 25
   :header-rows: 1

   * - Use Case
     - Recommended Basis
     - Why
     - Example
   * - General-purpose
     - Chebyshev1st, Legendre
     - Stable, well-conditioned
     - Classification, regression
   * - Smooth functions
     - Jacobi, Gegenbauer
     - Flexible weight functions
     - Physics-informed NNs
   * - Periodic patterns
     - Chebyshev (all kinds)
     - Trigonometric formulation
     - Signal processing
   * - Positive inputs
     - GeneralizedLaguerre
     - Natural domain [0, ∞)
     - Time series, survival
   * - Unbounded inputs
     - Hermite
     - Defined on ℝ
     - Gaussian-like data
   * - Local features
     - RBFs (Gaussian, etc.)
     - Localized response
     - Interpolation
   * - Multi-scale
     - Wavelets (Ricker, etc.)
     - Time-frequency analysis
     - Anomaly detection

Polynomial Bases
----------------

Orthogonal Polynomials (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Chebyshev (1st kind)** — ``Chebyshev1st``
  Best for general use. Minimal Runge phenomenon, excellent conditioning.
  
  - Domain: ``[-1, 1]``
  - Use ``input_clip=(-1, 1)`` for inputs outside this range
  - Degree: 3-10 for most tasks

**Legendre** — ``Legendre``
  Similar to Chebyshev, uniform weight function.
  
  - Domain: ``[-1, 1]``
  - Good for physics-based problems
  - Slightly less numerically stable than Chebyshev at high degrees

**Chebyshev (2nd, 3rd, 4th kinds)** — ``Chebyshev2nd``, ``Chebyshev3rd``, ``Chebyshev4th``
  Variants with different boundary behavior.
  
  - Use 2nd kind when you need ``sin``-like behavior at boundaries
  - 3rd/4th kinds are less common but useful for specific spectral methods

**Jacobi** — ``Jacobi``
  Generalization with two parameters (α, β).
  
  - Domain: ``[-1, 1]``
  - Set ``alpha_init``, ``beta_init`` based on your weight function
  - Chebyshev/Legendre/Gegenbauer are special cases

**Gegenbauer** — ``Gegenbauer``
  Single-parameter generalization (α > -1/2).
  
  - Domain: ``[-1, 1]``
  - α = 0: Chebyshev, α = 0.5: Legendre
  - Good when you need parameter flexibility

**Generalized Laguerre** — ``GeneralizedLaguerre``
  For positive-valued inputs.
  
  - Domain: ``[0, ∞)``
  - Use ``input_clip=(0, None)`` or ensure inputs are positive
  - Good for exponentially decaying functions

**Hermite** — ``Hermite``
  For unbounded inputs with Gaussian-like distributions.
  
  - Domain: ``ℝ``
  - Use ``normalized=True`` for probabilist's Hermite (better conditioned)
  - Warning: grows rapidly at high degree

Non-Orthogonal Polynomials
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Boubaker** — ``Boubaker``
  Specialized polynomial with unique recurrence.
  
  - Domain: ``ℝ``
  - Less common, but useful for specific physical models

**Fibonacci-like** — ``Fibonacci``, ``Lucas``, ``Pell``, etc.
  Integer sequence-based polynomials.
  
  - Domain: ``ℝ``
  - Novel architectures, experimental

**Laurent** — ``Laurent``
  Includes negative powers: ``∑ aₖ xᵏ`` for ``k ∈ ℤ``
  
  - Domain: ``ℝ \ {0}``
  - Internal clamp avoids poles near zero
  - Use when data has reciprocal relationships

Radial Basis Functions
----------------------

**Gaussian RBF** — ``GaussianRBF``
  Most common choice. Smooth, infinitely differentiable.
  
  - ``ε`` controls width (trainable by default)
  - Good for interpolation, smooth function approximation

**Multiquadric / Inverse Multiquadric** — ``MultiquadricRBF``, ``InverseMultiQuadricRBF``
  Less localized than Gaussian.
  
  - Multiquadric: grows with distance (use with care)
  - Inverse: decays, bounded

**Inverse Quadric / Cauchy** — ``InverseQuadricRBF``, ``CauchyRBF``
  Heavy-tailed decay.
  
  - More robust to outliers
  - Slower decay than Gaussian

**Thin Plate Spline** — ``ThinPlateSplineRBF``
  ``φ(r) = r² ln(r)``
  
  - Classic for scattered data interpolation
  - Not strictly positive definite

**Power / Linear / Cubic** — ``PowerRBF``, ``LinearRBF``, ``CubicRBF``
  Simple polynomial kernels.
  
  - ``PowerRBF``: ``r^p`` with trainable ``p``
  - ``CubicRBF``: ``r³``
  - ``LinearRBF``: ``r``

Wavelets
--------

**Ricker (Mexican Hat)** — ``Ricker``
  Second derivative of Gaussian. Zero mean.
  
  - ``σ`` controls scale (trainable)
  - Good for edge/peak detection

**Morlet (Gabor)** — ``Morelet``
  Gaussian-modulated sinusoid.
  
  - ``ω`` controls frequency (trainable)
  - Time-frequency analysis

**Derivative of Gaussian** — ``DerivativeOfGaussian``
  First derivative of Gaussian.
  
  - Antisymmetric
  - Edge detection

**Meyer** — ``Meyer``
  Compact frequency support.
  
  - Orthogonal wavelet
  - Good frequency localization

**Shannon** — ``Shannon``
  Ideal bandpass filter.
  
  - ``sinc``-based
  - Perfect frequency localization (infinite time support)

**Bump** — ``Bump``
  Compact support in time domain.
  
  - Smooth cutoff at boundaries
  - Good for local features

**Poisson** — ``Poisson``
  Cauchy-like wavelet.
  
  - Heavy tails
  - Robust to outliers

Choosing Hyperparameters
------------------------

Degree Selection (Polynomials)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Degree  | Capacity | Stability | Recommendation
    --------|----------|-----------|---------------
    1-3     | Low      | Excellent | Simple relationships
    4-8     | Medium   | Good      | Most tasks (default: 5)
    9-15    | High     | Fair      | Complex functions, use Clenshaw
    16+     | Very High| Poor      | Avoid unless necessary

Grid/Scale Selection (RBFs/Wavelets)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **RBF num_grids**: 5-20 typically; more grids = finer resolution but more parameters
- **RBF epsilon**: 0.1-10.0 typically; smaller = wider basis functions
- **Wavelet scale**: 0.5-5.0 typically; larger = coarser features

Hardware Considerations
-----------------------

.. list-table:: Hardware-Optimized Defaults
   :widths: 20 20 30 30
   :header-rows: 1

   * - Hardware
     - Precision
     - High Degree (>10)
     - Recommendation
   * - CPU
     - float64 OK
     - Use Clenshaw + float64
     - ``promote_to_float64=True``
   * - GPU (CUDA)
     - float32 optimal
     - Use Clenshaw
     - Keep float32
   * - Apple MPS
     - float32 only
     - Limit degree < 12
     - Use stable bases
   * - TPU
     - bfloat16/float32
     - Avoid high degree
     - Use simple bases

Mixed-Precision Training
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import tensorflow as tf
    from arnold.layers import Legendre
    
    # Enable mixed precision
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # RBFs and wavelets work well with mixed precision
    # For high-degree polynomials, consider:
    layer = Legendre(degree=12, units=32, promote_to_float64=True)

Common Pitfalls
---------------

1. **Inputs outside basis domain**
   
   Always use ``input_clip`` or preprocess:
   
   .. code-block:: python
   
       # Wrong: inputs may be outside [-1, 1]
       layer = Chebyshev1st(degree=5, units=10)
       
       # Correct: clip inputs
       layer = Chebyshev1st(degree=5, units=10, input_clip=(-1, 1))

2. **Too high degree**
   
   High degrees cause numerical instability:
   
   .. code-block:: python
   
       # Risky: may overflow
       layer = Hermite(degree=20, units=10)
       
       # Better: use normalized form + float64
       layer = Hermite(degree=20, units=10, normalized=True, promote_to_float64=True)

3. **Wrong RBF grid range**
   
   Match grid to your data range:
   
   .. code-block:: python
   
       # If your data is in [0, 100]
       layer = GaussianRBF(units=10, grid_min=0.0, grid_max=100.0, num_grids=20)

4. **Ignoring parameter constraints**
   
   Some polynomials require specific parameter ranges:
   
   .. code-block:: python
   
       # Gegenbauer: α > -0.5
       layer = Gegenbauer(degree=5, units=10, alpha_init=0.5)
       
       # Jacobi: α, β > -1
       layer = Jacobi(degree=5, units=10, alpha_init=0.5, beta_init=-0.5)
