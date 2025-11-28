.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _rbf-kan-layers:

================================================
Radial Basis Function KAN Layers
================================================

This chapter provides the complete API reference for radial basis function (RBF) 
KAN layers in ARNOLD. RBF-KANs use localized basis functions centered at grid points,
offering excellent interpolation properties and local adaptivity.

.. contents:: In This Chapter
   :local:
   :depth: 2

----

Overview
--------

Radial Basis Function KAN layers approximate univariate functions as:

.. math::

   \phi(x) = \sum_{k=1}^{K} c_k \, \psi\left(\frac{x - \mu_k}{\gamma_k}\right)

where:

- :math:`\psi` is the radial basis function
- :math:`\mu_k` are the grid centers
- :math:`\gamma_k` are the width (scale) parameters
- :math:`c_k` are learnable coefficients

**Key advantages:**

- **Local interpolation** — each basis affects only nearby points
- **Meshfree approximation** — no structured grid required
- **Universal approximation** — dense in :math:`C(\mathbb{R}^n)`
- **Smooth derivatives** — for most RBF types

**When to choose RBF KANs:**

- Scattered data interpolation
- Unknown or complex function domains
- Need for localized basis influence
- Meshfree methods and point cloud processing

**Quick comparison:**

.. list-table::
   :widths: 25 20 30 25
   :header-rows: 1

   * - RBF Type
     - Formula
     - Properties
     - Best For
   * - Gaussian
     - :math:`e^{-r^2}`
     - Infinitely smooth, local
     - General use
   * - Multiquadric
     - :math:`\sqrt{1+r^2}`
     - Global, smooth
     - Large-scale problems
   * - Thin-plate spline
     - :math:`r^2 \log r`
     - Minimizes curvature
     - Surface fitting

----

Stability and Performance Notes
-------------------------------

**Numerical Stability:**

- Radii are normalized by the grid spacing with a small :math:`\varepsilon` floor from 
  ``arnold.utils.constants.PARAM_EPS`` to avoid division by zero.
- Kernel shape parameters (``epsilon``, ``sigma``) are stored as logits and mapped 
  via ``softplus``; avoid extremely small inits that collapse kernels.

**Performance:**

- Basis evaluation uses the shared ``kan_function`` decorator
- Disable ``jit_compile`` via that decorator if you hit hardware-specific XLA issues
- Keep inputs within the expected data range or use ``input_clip`` on the layer 
  to prevent exploding radii

**Grid Design:**

- Default grid is uniform on ``[grid_min, grid_max]``
- Number of grid points (``num_grids``) controls approximation resolution
- More grid points = more parameters but finer resolution

----

Base Class
----------

All RBF KAN layers inherit from ``RBFBase``:

.. autoclass:: arnold.layers.core.radial_basis_functions.RBFBase
    :members: __init__, build, call
    :show-inheritance:

**Architecture:**

.. code-block:: text

   Input x ∈ ℝ^n
       │
       ▼
   ┌────────────────────┐
   │  Compute Distances │  r_k = (x - μ_k) / γ_k
   │    to Grid Centers │
   └────────────────────┘
       │
       ▼
   ┌────────────────────┐
   │   Evaluate RBF     │  ψ(r_k) for each center
   │   Φ ∈ ℝ^{n×K}      │
   └────────────────────┘
       │
       ▼
   ┌────────────────────┐
   │  Weight Tensor     │  Multiply with coefficients
   │   W ∈ ℝ^{n×m×K}    │
   └────────────────────┘
       │
       ▼
   Output y ∈ ℝ^m

----

Strictly Positive Definite RBFs
-------------------------------

These RBFs generate positive definite interpolation matrices, guaranteeing
unique interpolants for any distinct point set.

**Gaussian RBF** :math:`\psi(r) = e^{-r^2}`

The most commonly used RBF. Infinitely smooth, strictly local.

.. autoclass:: arnold.layers.core.radial_basis_functions.GaussianRBF
    :members: __init__

**Exponential RBF** :math:`\psi(r) = e^{-|r|}`

Less smooth than Gaussian, produces ridge-like features.

.. autoclass:: arnold.layers.core.radial_basis_functions.ExponentialRBF
    :members: __init__

**Inverse Quadric RBF** :math:`\psi(r) = \frac{1}{1+r^2}`

Bounded, smooth, decays polynomially.

.. autoclass:: arnold.layers.core.radial_basis_functions.InverseQuadricRBF
    :members: __init__

**Inverse Multiquadric RBF** :math:`\psi(r) = \frac{1}{\sqrt{1+r^2}}`

Global but bounded, good for scattered data.

.. autoclass:: arnold.layers.core.radial_basis_functions.InverseMultiQuadricRBF
    :members: __init__

**Cauchy RBF** :math:`\psi(r) = \frac{1}{1+r^2}`

Lorentzian shape, heavy-tailed influence.

.. autoclass:: arnold.layers.core.radial_basis_functions.CauchyRBF
    :members: __init__

----

Conditionally Positive Definite RBFs
------------------------------------

These require polynomial augmentation for well-posed interpolation but
offer excellent approximation properties.

**Multiquadric RBF** :math:`\psi(r) = \sqrt{1+r^2}`

Excellent for large-scale interpolation. Global influence.

.. autoclass:: arnold.layers.core.radial_basis_functions.MultiquadricRBF
    :members: __init__

**Thin-Plate Spline RBF** :math:`\psi(r) = r^2 \log r`

Minimizes bending energy. Natural for surface fitting.

.. autoclass:: arnold.layers.core.radial_basis_functions.ThinPlateSplineRBF
    :members: __init__

**Cubic RBF** :math:`\psi(r) = r^3`

Simple polynomial RBF, conditionally positive definite.

.. autoclass:: arnold.layers.core.radial_basis_functions.CubicRBF
    :members: __init__

**Linear RBF** :math:`\psi(r) = r`

Simplest conditionally positive definite RBF.

.. autoclass:: arnold.layers.core.radial_basis_functions.LinearRBF
    :members: __init__

**Power RBF** :math:`\psi(r) = r^p`

Generalized polynomial RBF with tunable power.

.. autoclass:: arnold.layers.core.radial_basis_functions.PowerRBF
    :members: __init__

----

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import tensorflow as tf
   from arnold.layers import GaussianRBF, MultiquadricRBF
   
   # Single layer with uniform grid
   layer = GaussianRBF(
       units=32,
       num_grids=16,    # 16 basis centers
       grid_min=-2.0,   # Grid lower bound
       grid_max=2.0,    # Grid upper bound
   )
   
   x = tf.random.uniform((16, 64), -2, 2)
   y = layer(x)  # Shape: (16, 32)

In a Model
~~~~~~~~~~

.. code-block:: python

   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(64,)),
       GaussianRBF(units=128, num_grids=20),
       tf.keras.layers.LayerNormalization(),
       GaussianRBF(units=64, num_grids=16),
       tf.keras.layers.LayerNormalization(),
       tf.keras.layers.Dense(10, activation='softmax'),
   ])

Mixed with Polynomial KANs
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arnold.layers import Chebyshev1st, GaussianRBF
   
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(100,)),
       
       # Polynomial for global structure
       Chebyshev1st(degree=8, units=64),
       tf.keras.layers.LayerNormalization(),
       
       # RBF for local features
       GaussianRBF(units=32, num_grids=24),
       tf.keras.layers.LayerNormalization(),
       
       tf.keras.layers.Dense(1),
   ])

Choosing Grid Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Coarse grid for smooth functions
   smooth_layer = GaussianRBF(units=32, num_grids=8)
   
   # Fine grid for complex functions
   complex_layer = GaussianRBF(units=32, num_grids=32)
   
   # Custom grid range for known data bounds
   bounded_layer = GaussianRBF(
       units=32,
       num_grids=16,
       grid_min=0.0,    # For positive data
       grid_max=10.0,
   )

----

Mathematical Details
--------------------

**Interpolation Theory:**

Given :math:`N` data points :math:`(x_i, y_i)`, RBF interpolation finds coefficients
:math:`c_k` such that:

.. math::

   s(x_i) = \sum_{k=1}^{N} c_k \psi(\|x_i - x_k\|) = y_i, \quad i = 1, \ldots, N

This leads to the linear system:

.. math::

   \Phi \mathbf{c} = \mathbf{y}

where :math:`\Phi_{ij} = \psi(\|x_i - x_j\|)`.

**Positive Definiteness:**

For strictly positive definite RBFs (Gaussian, inverse multiquadric), :math:`\Phi`
is positive definite for any distinct point set, guaranteeing a unique solution.

**Approximation Error:**

For smooth RBFs like Gaussian with fill distance :math:`h`, the error satisfies:

.. math::

   \|f - s_f\|_\infty = O(e^{-c/h})

exhibiting spectral convergence as the grid becomes denser.

----

See Also
--------

- :doc:`polynomial_kan_layers` — Polynomial-based alternatives
- :doc:`wavelet_kan_layers` — Multi-scale alternatives
- :doc:`../basis_selection_guide` — Choosing the right basis
- :doc:`../theory/approximation_theory` — Theoretical foundations
