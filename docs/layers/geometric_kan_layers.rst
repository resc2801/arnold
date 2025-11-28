.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _geometric-kan-layers:

====================================================
Geometric KAN Layers
====================================================

Basis functions for geometric and spherical domains.

Overview
--------

Geometric layers provide basis functions defined on spheres, disks, and 
higher-dimensional manifolds. They are essential for:

- Optics and aberration modeling (Zernike)
- Spherical harmonics expansions
- Computer graphics and 3D modeling
- Quantum mechanics angular momentum

Available Layers
----------------

Zernike
~~~~~~~

Zernike polynomials on the unit disk, widely used in optics.

.. code-block:: python

   from arnold.layers import Zernike
   
   layer = Zernike(
       units=32,
       degree=6,    # Maximum radial degree
   )

**Mathematical Definition:**

Zernike radial polynomials :math:`R_n^m(\rho)`:

.. math::

   R_n^m(\rho) = \sum_{k=0}^{(n-m)/2} 
   \frac{(-1)^k (n-k)!}{k! ((n+m)/2 - k)! ((n-m)/2 - k)!} \rho^{n-2k}

For :math:`m = 0` (radially symmetric):

.. math::

   R_n^0(\rho) = P_n(2\rho^2 - 1)

where :math:`P_n` is a Legendre polynomial.

**Parameters:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``degree``
     - Required
     - Maximum radial degree :math:`n`
   * - ``input_clip``
     - None
     - Clip inputs (use for unit disk constraint)

**Use Cases:**

- Optical aberration modeling
- Wavefront sensing
- Image feature extraction on circular domains


SphericalHarmonics
~~~~~~~~~~~~~~~~~~

Spherical harmonics for functions on the sphere :math:`S^2`.

.. code-block:: python

   from arnold.layers import SphericalHarmonics
   
   layer = SphericalHarmonics(
       units=32,
       degree=4,    # Maximum degree l
   )

**Mathematical Definition:**

Real spherical harmonics via Legendre polynomials:

.. math::

   Y_l^0(\theta) = N_l^0 P_l(\cos\theta)

where :math:`P_l` are Legendre polynomials and :math:`N_l^0` is a normalization
constant.

**Parameters:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``degree``
     - Required
     - Maximum harmonic degree :math:`l`

**Use Cases:**

- 3D graphics and rendering
- Geophysics and climate modeling
- Quantum angular momentum


HypersphericalHarmonics
~~~~~~~~~~~~~~~~~~~~~~~

Gegenbauer-based harmonics for :math:`n`-dimensional spheres :math:`S^{n-1}`.

.. code-block:: python

   from arnold.layers import HypersphericalHarmonics
   
   layer = HypersphericalHarmonics(
       units=32,
       degree=4,
       dimension=4,  # Sphere S^{n-1} in R^n
   )

**Mathematical Definition:**

For the :math:`n`-sphere, uses Gegenbauer polynomials :math:`C_l^{\alpha}`:

.. math::

   H_l^{(n)}(\cos\theta) = C_l^{(n-2)/2}(\cos\theta)

**Parameters:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``degree``
     - Required
     - Maximum harmonic degree
   * - ``dimension``
     - Required
     - Dimension :math:`n` of the embedding space

**Use Cases:**

- Higher-dimensional rotational symmetry
- Machine learning on manifolds
- Generalized angular momentum


Registry Access
---------------

.. code-block:: python

   from arnold.layers.core.registry import get_layer, list_layers_by_category
   
   # Get layer by name
   layer = get_layer("zernike", units=32, degree=4)
   
   # List all geometric layers
   geometric_layers = list_layers_by_category()["geometric"]

**Aliases:**

- ``zernike``, ``zernike_radial`` → Zernike
- ``spherical``, ``spherical_harmonics`` → SphericalHarmonics
- ``hyperspherical``, ``hyperspherical_harmonics`` → HypersphericalHarmonics


Performance Considerations
--------------------------

All geometric layers use stable Clenshaw-style recurrences for polynomial
evaluation and are fully XLA-compatible.

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Layer
     - Basis Count
     - GPU Efficiency
     - TPU Efficiency
   * - Zernike
     - O(d)
     - ✓✓✓
     - ✓✓✓
   * - SphericalHarmonics
     - O(l+1)
     - ✓✓✓
     - ✓✓✓
   * - HypersphericalHarmonics
     - O(l+1)
     - ✓✓✓
     - ✓✓✓

