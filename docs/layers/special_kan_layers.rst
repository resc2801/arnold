.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _special-kan-layers:

====================================================
Special Function KAN Layers
====================================================

Basis functions from special mathematical functions.

Overview
--------

Special function layers use classical special functions as basis:

- Airy functions (quantum mechanics, optics)
- Bessel functions (cylindrical symmetry)
- And more (Mathieu, Whittaker, etc.)

These are ideal for physics-informed neural networks where the underlying
physics naturally involves these special functions.

Available Layers
----------------

Airy
~~~~

Airy functions Ai(x) and Bi(x) as basis.

.. code-block:: python

   from arnold.layers import Airy
   
   layer = Airy(
       units=32,
       num_basis=8,     # Number of Airy function shifts
   )

**Mathematical Definition:**

The Airy function Ai(x) is defined as:

.. math::

   \text{Ai}(x) = \frac{1}{\pi} \int_0^\infty \cos\left(\frac{t^3}{3} + xt\right) dt

It is the solution to the Airy differential equation:

.. math::

   y'' - xy = 0

**Parameters:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``num_basis``
     - 8
     - Number of shifted Airy functions in basis

**Use Cases:**

- Quantum tunneling problems
- Rainbow optics
- Caustics in wave propagation


BesselFunctions
~~~~~~~~~~~~~~~

Bessel functions of the first kind as basis.

.. code-block:: python

   from arnold.layers import BesselFunctions
   
   layer = BesselFunctions(
       units=32,
       max_order=6,     # Maximum Bessel order
   )

**Mathematical Definition:**

Bessel function of the first kind :math:`J_\nu(x)`:

.. math::

   J_\nu(x) = \sum_{m=0}^\infty \frac{(-1)^m}{m! \Gamma(m + \nu + 1)}
   \left(\frac{x}{2}\right)^{2m + \nu}

**Parameters:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``max_order``
     - 4
     - Maximum Bessel order :math:`\nu = 0, 1, \ldots, \text{max\_order}`

**Use Cases:**

- Cylindrical wave problems
- Vibrating membranes
- Heat conduction in cylinders


Future Layers (Stubs)
---------------------

The following layers are planned but not yet fully implemented:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Layer
     - Description
   * - ParabolicCylinder
     - Parabolic cylinder functions for quantum harmonic oscillator
   * - Mathieu
     - Mathieu functions for elliptic coordinate systems
   * - Whittaker
     - Whittaker functions (confluent hypergeometric)
   * - Slepian
     - Prolate spheroidal wave functions
   * - LegendreFunctions
     - Associated Legendre functions (beyond polynomials)
   * - EllipticFunctions
     - Jacobi elliptic functions

These are available as stub implementations that raise ``NotImplementedError``
when called, preserving the API for future development.


Registry Access
---------------

.. code-block:: python

   from arnold.layers.core.registry import get_layer, list_layers_by_category
   
   # Get layer by name
   layer = get_layer("airy", units=32)
   layer = get_layer("bessel_functions", units=32, max_order=4)
   
   # List all special function layers
   special_layers = list_layers_by_category()["special"]

**Aliases:**

- ``airy``, ``airy_function`` → Airy
- ``bessel``, ``bessel_j``, ``bessel_functions`` → BesselFunctions


Mathematical Background
-----------------------

Special functions arise as solutions to important differential equations:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Function
     - Differential Equation
   * - Airy Ai, Bi
     - :math:`y'' - xy = 0`
   * - Bessel :math:`J_\nu`
     - :math:`x^2 y'' + x y' + (x^2 - \nu^2) y = 0`
   * - Hermite
     - :math:`y'' - 2xy' + 2ny = 0`
   * - Laguerre
     - :math:`xy'' + (1-x)y' + ny = 0`

Using these as basis functions in KANs can provide **inductive bias** for
problems where the physics naturally involves these functions.

