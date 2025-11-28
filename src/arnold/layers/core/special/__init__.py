# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Special function KAN layers.

This subpackage provides KAN layers based on special mathematical functions,
including classical solutions of differential equations and transcendental
functions with important applications in physics and engineering.

Available Layers
----------------

**Differential Equation Solutions:**

.. autosummary::
    :toctree: generated/

    Airy          - Solutions of Airy's equation y'' = xy
    Bessel        - Solutions of Bessel's differential equation
    ParabolicCylinder - Parabolic cylinder functions D_ν(x)
    Mathieu       - Solutions of Mathieu's equation
    Whittaker     - Whittaker functions M_κ,μ(x), W_κ,μ(x)

**Signal Processing:**

.. autosummary::
    :toctree: generated/

    Slepian       - Discrete prolate spheroidal sequences (DPSS)

**Other Special Functions:**

.. autosummary::
    :toctree: generated/

    LegendreFunctions  - Legendre functions P_ν^μ(x), Q_ν^μ(x)
    EllipticFunctions  - Elliptic functions (Jacobi elliptic, etc.)

Mathematical Background
-----------------------
Special functions arise naturally as solutions to important differential
equations in mathematical physics:

- **Airy functions**: :math:`y'' = xy` (turning points in wave mechanics)
- **Bessel functions**: :math:`x^2 y'' + x y' + (x^2 - ν^2)y = 0` (cylindrical symmetry)
- **Parabolic cylinder**: :math:`y'' + (ν + 1/2 - x^2/4)y = 0` (quantum harmonic oscillator)
- **Mathieu functions**: :math:`y'' + (a - 2q\cos 2x)y = 0` (elliptic membranes)

These functions provide optimal bases for problems with the corresponding
symmetries or boundary conditions.

Usage Example
-------------
>>> import tensorflow as tf
>>> from arnold.layers.core.special import Airy, Bessel
>>>
>>> # Airy function basis for turning point problems
>>> layer = Airy(max_degree=8, units=32)
>>> x = tf.random.uniform((16, 10), -5, 5)
>>> y = layer(x)
>>>
>>> # Bessel function basis for cylindrical problems
>>> layer = Bessel(max_order=6, units=64)
>>> r = tf.random.uniform((16, 10), 0, 10)
>>> y = layer(r)

References
----------
.. [1] Abramowitz, M. & Stegun, I.A. (1972). "Handbook of Mathematical Functions"
.. [2] NIST Digital Library of Mathematical Functions. https://dlmf.nist.gov/
.. [3] Olver, F.W.J. (2010). "NIST Handbook of Mathematical Functions"
"""

from arnold.layers.core.special.airy import Airy
from arnold.layers.core.special.base import SpecialBase
from arnold.layers.core.special.bessel_functions import Bessel
from arnold.layers.core.special.elliptic_functions import EllipticFunctions
from arnold.layers.core.special.legendre_functions import LegendreFunctions
from arnold.layers.core.special.mathieu import Mathieu
from arnold.layers.core.special.parabolic_cylinder import ParabolicCylinder
from arnold.layers.core.special.slepian import Slepian
from arnold.layers.core.special.whittaker import Whittaker


__all__ = [
    # Base class
    "SpecialBase",
    # Differential equation solutions
    "Airy",
    "Bessel",
    "ParabolicCylinder",
    "Mathieu",
    "Whittaker",
    # Signal processing
    "Slepian",
    # Other special functions
    "LegendreFunctions",
    "EllipticFunctions",
]
