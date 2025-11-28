.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _polynomial-kan-layers:

================================================
Polynomial KAN Layers
================================================

This chapter provides the complete API reference for all polynomial-based KAN layers
in ARNOLD. Polynomial KANs use orthogonal polynomial bases to learn univariate functions
:math:`\phi_{q,p}(x_p)` with strong approximation-theoretic guarantees.

.. contents:: In This Chapter
   :local:
   :depth: 2

----

Overview
--------

Polynomial KAN layers are the most mathematically well-understood family in ARNOLD.
They offer:

- **Spectral convergence** for smooth functions
- **Optimal approximation rates** from classical approximation theory
- **Efficient computation** via three-term recurrence relations
- **Rich mathematical structure** from orthogonal polynomial theory

**When to choose Polynomial KANs:**

- Target function is smooth (differentiable)
- Physical systems with known structure (Legendre for physics, Hermite for quantum)
- Need for interpretable basis coefficients
- Known function domain (polynomials have natural domains)

**Quick comparison:**

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Polynomial
     - Domain
     - Best For
     - Convergence
   * - Chebyshev 1st
     - :math:`[-1, 1]`
     - General use
     - Spectral
   * - Legendre
     - :math:`[-1, 1]`
     - Physics, PDEs
     - Spectral
   * - Hermite
     - :math:`\mathbb{R}`
     - Gaussian data
     - Spectral
   * - Laguerre
     - :math:`[0, \infty)`
     - Decay/lifetimes
     - Spectral

----

Base Class
----------

All polynomial KAN layers inherit from ``PolynomialBase``:

.. autoclass:: arnold.layers.core.polynomial.PolynomialBase
    :members: __init__, build, call, pseudo_vandermonde
    :show-inheritance:

**Architecture:**

.. code-block:: text

   Input x ∈ ℝ^n
       │
       ▼
   ┌────────────────────┐
   │ Pseudo-Vandermonde │  Compute P_k(x_i) for k=0,...,d
   │   V(x) ∈ ℝ^{n×(d+1)}│
   └────────────────────┘
       │
       ▼
   ┌────────────────────┐
   │  Alternant Tensor  │  Multiply with coefficients c_{i,j,k}
   │   W ∈ ℝ^{n×m×(d+1)}│
   └────────────────────┘
       │
       ▼
   Output y ∈ ℝ^m

**Creating Custom Polynomial Layers:**

To implement a custom polynomial basis, override ``compute_polynomials``:

.. code-block:: python

   from arnold.layers.core.polynomial import PolynomialBase
   
   class MyPolynomial(PolynomialBase):
       def compute_polynomials(self, x):
           \"\"\"Return shape (..., input_dim, degree+1).\"\"\"
           # Implement your recurrence relation here
           pass

----

Orthogonal Polynomial Bases
---------------------------

Orthogonal polynomials satisfy:

.. math::

   \int_a^b P_n(x) P_m(x) w(x) \, dx = h_n \delta_{nm}

where :math:`w(x)` is the weight function. This orthogonality ensures:

- **Numerical stability** in coefficient computation
- **Uncorrelated basis functions** for efficient learning
- **Optimal approximation** via least-squares projection

Classical Orthogonal Polynomials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Chebyshev Polynomials of the First Kind** :math:`T_n(x)`

Best default choice for general function approximation.

.. math::

   T_n(x) = \cos(n \arccos x), \quad x \in [-1, 1]

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Chebyshev1st
    :members: __init__
    :noindex:

**Chebyshev Polynomials of the Second Kind** :math:`U_n(x)`

.. math::

   U_n(x) = \frac{\sin((n+1) \arccos x)}{\sin(\arccos x)}

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Chebyshev2nd
    :members: __init__
    :noindex:

**Chebyshev Polynomials of the Third Kind** :math:`V_n(x)`

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Chebyshev3rd
    :members: __init__

**Chebyshev Polynomials of the Fourth Kind** :math:`W_n(x)`

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Chebyshev4th
    :members: __init__

**Legendre Polynomials** :math:`P_n(x)`

Optimal for physical problems with uniform weight.

.. math::

   P_n(x) = \frac{1}{2^n n!} \frac{d^n}{dx^n}[(x^2 - 1)^n]

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Legendre
    :members: __init__

**Hermite Polynomials** :math:`H_n(x)` (Physicist's)

Natural for Gaussian-weighted data and quantum mechanics.

.. math::

   H_n(x) = (-1)^n e^{x^2} \frac{d^n}{dx^n} e^{-x^2}

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Hermite
    :members: __init__

**Generalized Laguerre Polynomials** :math:`L_n^{(\alpha)}(x)`

For data on :math:`[0, \infty)` with exponential decay.

.. math::

   L_n^{(\alpha)}(x) = \frac{x^{-\alpha} e^x}{n!} \frac{d^n}{dx^n}(e^{-x} x^{n+\alpha})

.. autoclass:: arnold.layers.core.polynomial.orthogonal.GeneralizedLaguerre
    :members: __init__

**Jacobi Polynomials** :math:`P_n^{(\alpha,\beta)}(x)`

Flexible two-parameter family including Chebyshev, Legendre, Gegenbauer.

.. math::

   w(x) = (1-x)^\alpha (1+x)^\beta, \quad \alpha, \beta > -1

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Jacobi
    :members: __init__

**Gegenbauer (Ultraspherical) Polynomials** :math:`C_n^{(\lambda)}(x)`

Special case of Jacobi with :math:`\alpha = \beta = \lambda - 1/2`.

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Gegenbauer
    :members: __init__

Discrete Orthogonal Polynomials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Charlier Polynomials**

Orthogonal w.r.t. Poisson distribution.

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Charlier
    :members: __init__

**Pollaczek Polynomials**

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Pollaczek
    :members: __init__

**Associated Meixner-Pollaczek Polynomials**

.. autoclass:: arnold.layers.core.polynomial.orthogonal.AssociatedMeixnerPollaczek
    :members: __init__

Advanced Orthogonal Families
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Al-Salam-Carlitz I Polynomials** :math:`U_n^{(a)}(x; q)`

:math:`q`-analogs of orthogonal polynomials.

.. autoclass:: arnold.layers.core.polynomial.orthogonal.AlSalamCarlitz1st
    :members: __init__

**Al-Salam-Carlitz II Polynomials** :math:`V_n^{(a)}(x; q)`

.. autoclass:: arnold.layers.core.polynomial.orthogonal.AlSalamCarlitz2nd
    :members: __init__

**Askey-Wilson Polynomials**

The most general classical orthogonal polynomials.

.. autoclass:: arnold.layers.core.polynomial.orthogonal.AskeyWilson
    :members: __init__

**Bannai-Ito Polynomials**

.. autoclass:: arnold.layers.core.polynomial.orthogonal.BannaiIto
    :members: __init__

**Bessel Polynomials**

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Bessel
    :members: __init__

**Wilson Polynomials**

.. autoclass:: arnold.layers.core.polynomial.orthogonal.Wilson
    :members: __init__

----

Discrete Orthogonal Polynomials
-------------------------------

Discrete orthogonal polynomials are orthogonal with respect to discrete measures
(sums over discrete points rather than integrals). They are particularly useful
for modeling discrete data, probability distributions, and combinatorial structures.

These polynomials form part of the Askey scheme and satisfy discrete orthogonality:

.. math::

   \sum_{x=0}^{N} w(x) P_m(x) P_n(x) = h_n \delta_{mn}

where :math:`w(x)` is a discrete weight function defined on a finite or countably
infinite set of points.

**Krawtchouk Polynomials** :math:`K_n(x; p, N)`

Orthogonal on :math:`\{0, 1, \ldots, N\}` with binomial weight. Used in coding theory,
probability (binomial distributions), and quantum mechanics.

.. math::

   K_n(x; p, N) = \sum_{j=0}^{n} \binom{n}{j} \binom{N-n}{x-j} (-1)^j \left(\frac{1-p}{p}\right)^j

.. autoclass:: arnold.layers.core.polynomial.discrete.Krawtchouk
    :members: __init__

**Hahn Polynomials** :math:`Q_n(x; \alpha, \beta, N)`

Generalization of Krawtchouk with two shape parameters. Orthogonal on
:math:`\{0, 1, \ldots, N\}` with hypergeometric weight.

.. math::

   w(x) = \binom{\alpha + x}{x} \binom{\beta + N - x}{N - x}, \quad \alpha, \beta > -1

.. autoclass:: arnold.layers.core.polynomial.discrete.Hahn
    :members: __init__

**Meixner Polynomials** :math:`M_n(x; \beta, c)`

Orthogonal on :math:`\{0, 1, 2, \ldots\}` with negative binomial weight.
Model waiting times and count data.

.. math::

   w(x) = \frac{(\beta)_x c^x}{x!}, \quad \beta > 0, \; 0 < c < 1

.. autoclass:: arnold.layers.core.polynomial.discrete.Meixner
    :members: __init__

**Racah Polynomials** :math:`R_n(\lambda(x); \alpha, \beta, \gamma, \delta)`

The most general classical discrete orthogonal polynomials (at the top of the
discrete part of the Askey scheme). Used in quantum angular momentum theory
(Racah coefficients / 6j-symbols).

.. math::

   \lambda(x) = x(x + \gamma + \delta + 1)

.. autoclass:: arnold.layers.core.polynomial.discrete.Racah
    :members: __init__

----

Non-Orthogonal Polynomial Bases
-------------------------------

Non-orthogonal polynomials offer specialized structure for particular applications.

**Boubaker Polynomials**

Useful in physics and engineering applications.

.. autoclass:: arnold.layers.core.polynomial.Boubaker
    :members: __init__

**Laurent Polynomials**

Allow negative powers, useful for periodic functions.

.. autoclass:: arnold.layers.core.polynomial.Laurent
    :members: __init__

----

Lucas Polynomial Sequences
--------------------------

Lucas polynomial sequences generalize Fibonacci-like recurrences:

.. math::

   W_n(x) = x \cdot W_{n-1}(x) + W_{n-2}(x)

These appear in algebraic number theory and combinatorics.

**Fermat Polynomials**

.. autoclass:: arnold.layers.core.polynomial.w_polynomials.Fermat
    :members: __init__

**Fermat-Lucas Polynomials**

.. autoclass:: arnold.layers.core.polynomial.w_polynomials.FermatLucas
    :members: __init__

**Jacobsthal Polynomials**

.. autoclass:: arnold.layers.core.polynomial.w_polynomials.Jacobsthal
    :members: __init__

**Jacobsthal-Lucas Polynomials**

.. autoclass:: arnold.layers.core.polynomial.w_polynomials.JacobsthalLucas
    :members: __init__

**Pell Polynomials**

.. autoclass:: arnold.layers.core.polynomial.w_polynomials.Pell
    :members: __init__

**Pell-Lucas Polynomials**

.. autoclass:: arnold.layers.core.polynomial.w_polynomials.PellLucas
    :members: __init__

----

Generalized Fibonacci Polynomials
---------------------------------

Higher-order Fibonacci-type recurrences for :math:`n`-bonacci sequences:

.. math::

   F_n^{(k)}(x) = x \cdot F_{n-1}^{(k)}(x) + x \cdot F_{n-2}^{(k)}(x) + \cdots + x \cdot F_{n-k}^{(k)}(x)

**Fibonacci Polynomials** (:math:`k=2`)

.. autoclass:: arnold.layers.core.polynomial.n_bonacci.Fibonacci
    :members: __init__

**Tribonacci Polynomials** (:math:`k=3`)

Not included yet—contributions welcome!

**Tetranacci Polynomials** (:math:`k=4`)

.. autoclass:: arnold.layers.core.polynomial.n_bonacci.Tetranacci
    :members: __init__

**Pentanacci Polynomials** (:math:`k=5`)

.. autoclass:: arnold.layers.core.polynomial.n_bonacci.Pentanacci
    :members: __init__

**Hexanacci Polynomials** (:math:`k=6`)

.. autoclass:: arnold.layers.core.polynomial.n_bonacci.Hexanacci
    :members: __init__

**Heptanacci Polynomials** (:math:`k=7`)

.. autoclass:: arnold.layers.core.polynomial.n_bonacci.Heptanacci
    :members: __init__

**Octanacci Polynomials** (:math:`k=8`)

.. autoclass:: arnold.layers.core.polynomial.n_bonacci.Octanacci
    :members: __init__

----

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import tensorflow as tf
   from arnold.layers import Chebyshev1st, Legendre, Hermite
   
   # Single layer
   layer = Chebyshev1st(degree=8, units=32)
   x = tf.random.uniform((16, 64), -1, 1)
   y = layer(x)  # Shape: (16, 32)
   
   # In a model
   model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(64,)),
       Chebyshev1st(degree=8, units=32),
       tf.keras.layers.LayerNormalization(),
       Legendre(degree=5, units=10),
   ])

With Tucker Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Memory-efficient for large dimensions
   layer = Legendre(
       degree=10,
       units=256,
       core_ranks=(32, 32, 8),  # Tucker decomposition
   )
   
   x = tf.random.uniform((16, 1024), -1, 1)
   y = layer(x)  # Shape: (16, 256)

Problem-Specific Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Physics/PDEs: Legendre (uniform weight)
   physics_layer = Legendre(degree=12, units=64)
   
   # Gaussian data: Hermite
   gaussian_layer = Hermite(degree=8, units=32)
   
   # Flexible: Jacobi with custom parameters
   custom_layer = Jacobi(degree=10, units=32, alpha=0.5, beta=-0.3)
   
   # Decay/lifetime data: Laguerre
   decay_layer = GeneralizedLaguerre(degree=8, units=32, alpha=1.0)

Discrete Orthogonal Polynomials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arnold.layers import Krawtchouk, Hahn, Meixner, Racah
   
   # Krawtchouk: discrete data with binomial structure
   # p in (0,1), N is support size
   layer = Krawtchouk(degree=5, units=32, p=0.5, N=10)
   
   # Hahn: more flexible discrete data
   # alpha, beta > -1, N is support size
   layer = Hahn(degree=5, units=32, alpha=1.0, beta=1.0, N=10)
   
   # Meixner: count data / negative binomial
   # beta > 0, c in (0,1)
   layer = Meixner(degree=5, units=32, beta=2.0, c=0.5)
   
   # Racah: most general discrete polynomials
   layer = Racah(degree=5, units=32, alpha=1.0, beta=1.0, gamma=0.5, delta=0.5, N=10)
   
   # With trainable parameters (learn optimal discrete distribution)
   layer = Krawtchouk(degree=5, units=32, trainable_params=True)

----

See Also
--------

- :doc:`../theory/approximation_theory` — Mathematical foundations
- :doc:`../basis_selection_guide` — Choosing the right polynomial
- :doc:`radial_basis_kan_layers` — RBF-based alternatives
- :doc:`wavelet_kan_layers` — Wavelet-based alternatives

----

References
----------

For detailed mathematical definitions and properties of orthogonal polynomials, 
see the NIST Digital Library of Mathematical Functions (DLMF) and Wikipedia.

- NIST Digital Library of Mathematical Functions, Chapter 18: https://dlmf.nist.gov/18
- Orthogonal polynomials: https://en.wikipedia.org/wiki/Orthogonal_polynomials