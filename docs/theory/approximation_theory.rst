.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _approximation_theory:

================================================
Approximation Theory for KANs
================================================

This chapter covers the theoretical foundations of function approximation that underpin 
Kolmogorov-Arnold Networks. We examine convergence rates, approximation errors, and the 
mathematical properties that make different basis functions suitable for different problems.

.. contents:: Chapter Contents
   :local:
   :depth: 2

----

Universal Approximation
-----------------------

Classical Results
~~~~~~~~~~~~~~~~~

The Kolmogorov-Arnold theorem guarantees that continuous functions can be **exactly** 
represented (not just approximated) using the KAN structure. However, this exact 
representation requires potentially non-smooth inner/outer functions.

In practice, we use smooth basis expansions (polynomials, RBFs, wavelets) which introduce 
**approximation error**. The key question is: how does this error decrease as we increase 
the basis complexity?

Weierstrass Approximation Theorem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Theorem - Weierstrass Approximation
   

   For any continuous function :math:`f: [a, b] \to \mathbb{R}` and :math:`\varepsilon > 0`, 
   there exists a polynomial :math:`p` such that:

   .. math::

      \sup_{x \in [a, b]} |f(x) - p(x)| < \varepsilon

This foundational result justifies the use of polynomial bases in KAN layers: polynomials 
can approximate any continuous univariate function to arbitrary accuracy.

Stone-Weierstrass Generalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Stone-Weierstrass theorem extends this to more general function spaces, including 
RBFs and wavelets under appropriate conditions.

----

Polynomial Approximation
------------------------

Jackson's Theorem (Direct)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Theorem - Jackson's Theorem
   

   Let :math:`f \in C^k[-1, 1]` (k-times continuously differentiable). Then the best 
   polynomial approximation of degree :math:`n` satisfies:

   .. math::

      E_n(f) := \inf_{\deg p \leq n} \|f - p\|_\infty = O\left(\frac{1}{n^k}\right)

   More precisely:

   .. math::

      E_n(f) \leq \frac{C_k}{n^k} \omega\left(f^{(k)}; \frac{1}{n}\right)

   where :math:`\omega(g; \delta)` is the modulus of continuity of :math:`g`.

**Implications for KANs:**

- Smoother target functions (:math:`k` large) converge faster
- Degree :math:`n = 10` gives :math:`O(10^{-k})` error for :math:`C^k` functions
- Polynomial degree directly controls approximation accuracy

Bernstein's Theorem (Inverse)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Theorem - Bernstein's Inverse Theorem
   

   If :math:`E_n(f) = O(1/n^k)` for all :math:`n`, then :math:`f \in C^{k-1}[-1, 1]` and 
   :math:`f^{(k-1)}` is Lipschitz.

This means fast polynomial convergence **implies** smoothness—a powerful diagnostic for 
understanding your data.

Convergence Rates by Basis
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Polynomial Basis Convergence
   :header-rows: 1
   :widths: 25 35 40

   * - Basis
     - Best For
     - Convergence Rate
   * - Chebyshev
     - Smooth on :math:`[-1,1]`
     - :math:`O(e^{-cn})` (analytic :math:`f`)
   * - Legendre
     - Uniform approximation
     - :math:`O(1/n^k)` for :math:`C^k`
   * - Hermite
     - Gaussian-weighted
     - :math:`O(e^{-cn^2})` for Gaussian decay
   * - Laguerre
     - Exponential decay on :math:`[0, \infty)`
     - :math:`O(1/n^k)` for appropriate weights

For **analytic functions** (infinitely differentiable with convergent Taylor series), 
Chebyshev approximation achieves **exponential** (spectral) convergence.

----

The Runge Phenomenon
--------------------

A cautionary tale for polynomial approximation:

**Why it matters for KANs:**

- High-degree polynomials with equidistant evaluation points can oscillate wildly
- **Solution 1**: Use Chebyshev nodes (clustered at endpoints)
- **Solution 2**: Use orthogonal polynomials (Chebyshev, Legendre) which are optimal
- **Solution 3**: Use RBFs or wavelets for local approximation

ARNOLD's orthogonal polynomial bases inherently avoid Runge-like instabilities because 
they use optimal approximation in the :math:`L^2` sense.

----

Orthogonal Polynomial Properties
--------------------------------

Why Orthogonality Matters
~~~~~~~~~~~~~~~~~~~~~~~~~

Orthogonal polynomials satisfy:

.. math::

   \int_a^b P_m(x) P_n(x) w(x) \, dx = h_n \delta_{mn}

where :math:`w(x)` is a weight function and :math:`h_n` is a normalization constant.

**Benefits:**

1. **Optimal approximation**: The orthogonal projection minimizes :math:`L^2` error
2. **Uncorrelated coefficients**: Each coefficient captures unique information
3. **Stable computation**: Three-term recurrences avoid catastrophic cancellation
4. **Fast coefficient computation**: Inner products are easy to compute

The Askey Scheme
~~~~~~~~~~~~~~~~

The **Askey scheme** classifies all classical orthogonal polynomials by their limit relations:

.. code-block:: text

                         Wilson
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        Continuous    Continuous    Continuous
         Dual Hahn    Hahn/Racah   dual Hahn
              │            │            │
              ▼            ▼            ▼
          Meixner-     Hahn         Krawtchouk
         Pollaczek                      │
              │            │            ▼
              ▼            ▼         Charlier
           Jacobi      Meixner
              │            │
         ┌────┴────┐       │
         ▼         ▼       ▼
     Gegenbauer  Laguerre  Poisson
         │
    ┌────┴────┐
    ▼         ▼
 Chebyshev  Legendre
    │
    ▼
 Hermite

ARNOLD provides layers for polynomials at all levels of this hierarchy.

Connection to Spectral Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

KAN layers with orthogonal polynomials are closely related to **spectral methods** in 
numerical analysis:

- **Galerkin methods**: Approximate solutions in polynomial subspaces
- **Collocation**: Enforce equations at specific points
- **Pseudospectral**: Combine pointwise evaluation with spectral accuracy

This connection suggests that KANs may be particularly effective for **physics-informed 
neural networks** (PINNs) where spectral accuracy is desirable.

----

RBF Approximation Theory
------------------------

Native Space
~~~~~~~~~~~~

Each RBF kernel :math:`\phi` has an associated **native space** :math:`\mathcal{N}_\phi` of 
functions for which the kernel provides optimal approximation.

.. admonition:: Theorem - RBF Error Bounds
   

   For :math:`f` in the native space of a positive definite RBF with :math:`N` centers, 
   the approximation error satisfies:

   .. math::

      \|f - s_f\|_\infty \leq C \, h^k \, |f|_{\mathcal{N}_\phi}

   where :math:`h` is the fill distance (maximum distance from any point to nearest center) 
   and :math:`k` depends on the smoothness of :math:`\phi`.

**Gaussian RBF**: Native space is a Sobolev space of infinite order (spectral convergence 
for analytic functions).

**Thin Plate Spline**: Native space is the Beppo-Levi space (algebraic convergence).

Shape Parameter Trade-off
~~~~~~~~~~~~~~~~~~~~~~~~~

The shape parameter :math:`\varepsilon` in Gaussian RBFs creates a trade-off:

- **Small** :math:`\varepsilon`: Flat, global kernels → better conditioning, but requires more centers
- **Large** :math:`\varepsilon`: Peaked, local kernels → fewer centers needed, but ill-conditioned

ARNOLD uses trainable :math:`\varepsilon` with softplus constraints to learn the optimal 
balance during training.

----

Wavelet Approximation
---------------------

Multiresolution Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

Wavelets provide **multiresolution approximation**: different scales capture different 
levels of detail.

.. admonition:: Theorem - Wavelet Characterization
   

   A function :math:`f \in L^2(\mathbb{R})` can be written as:

   .. math::

      f(x) = \sum_{j=-\infty}^{\infty} \sum_{k=-\infty}^{\infty} c_{j,k} \, \psi_{j,k}(x)

   where :math:`\psi_{j,k}(x) = 2^{j/2} \psi(2^j x - k)` are translated and dilated wavelets.

   The approximation at scale :math:`J` is:

   .. math::

      f_J(x) = \sum_{j=-\infty}^{J} \sum_k c_{j,k} \psi_{j,k}(x)

**Error decay**: For :math:`f` with :math:`r` bounded derivatives:

.. math::

   \|f - f_J\|_2 = O(2^{-Jr})

Localization Properties
~~~~~~~~~~~~~~~~~~~~~~~

Wavelets are characterized by their **time-frequency localization**:

.. list-table:: Wavelet Localization
   :header-rows: 1
   :widths: 25 35 40

   * - Wavelet
     - Time Localization
     - Frequency Localization
   * - Haar
     - Compact (best)
     - Poor
   * - Ricker (Mexican Hat)
     - Good
     - Good
   * - Morlet
     - Good
     - Good
   * - Shannon
     - Poor
     - Perfect (compact)
   * - Meyer
     - Good
     - Compact (best smooth)

For KAN layers, **Ricker** and **Morlet** provide good balance for general use.

----

Expressivity of KAN Architectures
---------------------------------

Width vs. Depth
~~~~~~~~~~~~~~~

The Kolmogorov-Arnold theorem uses width :math:`2n + 1` and depth 2. For practical KANs:

**Single Layer (Shallow)**:
   - Can approximate any continuous function (universal)
   - May require very high polynomial degree for complex functions
   - Risk of overfitting with many parameters

**Multiple Layers (Deep)**:
   - Hierarchical feature extraction
   - Lower degree per layer often suffices
   - Better generalization in practice

.. code-block:: python

   # Deep KAN: multiple layers with moderate degree
   model = tf.keras.Sequential([
       Chebyshev1st(degree=5, units=64),
       Chebyshev1st(degree=5, units=32),
       Chebyshev1st(degree=5, units=10),
   ])

Comparison with MLPs
~~~~~~~~~~~~~~~~~~~~

For approximating a function :math:`f: \mathbb{R}^n \to \mathbb{R}`:

- **MLP**: Requires :math:`O(\varepsilon^{-n/k})` parameters for :math:`\varepsilon`-approximation 
  of :math:`C^k` functions (curse of dimensionality)

- **KAN** (theoretical): Fixed structure with :math:`2n+1` width suffices for exact 
  representation (no curse)

- **KAN** (practical with polynomials): :math:`O(\varepsilon^{-1/k})` degree per dimension 
  (milder curse)

The key insight is that KANs can exploit **additive structure** in the target function, 
while MLPs must learn it implicitly.

----

Regularization and Generalization
---------------------------------

Bias-Variance Trade-off
~~~~~~~~~~~~~~~~~~~~~~~

Higher polynomial degrees increase **expressivity** but also **variance**:

.. math::

   \text{Error} = \underbrace{\text{Bias}^2}_{\text{decreases with degree}} + 
                  \underbrace{\text{Variance}}_{\text{increases with degree}} + 
                  \text{Noise}

**Recommendations:**

- Start with low degree (3-5) and increase if underfitting
- Use Tucker decomposition to limit effective capacity
- Add regularization for high degrees

Coefficient Regularization
~~~~~~~~~~~~~~~~~~~~~~~~~~

ARNOLD supports Keras regularizers on polynomial coefficients:

.. code-block:: python

   layer = Legendre(
       degree=10,
       units=64,
       kernel_regularizer=tf.keras.regularizers.L2(1e-4)
   )

:math:`L^2` regularization on coefficients corresponds to smoothness penalties on the 
learned functions :math:`f_{i,j}`.

----

Summary: Choosing Basis Functions
---------------------------------

Based on approximation theory:

.. list-table:: Basis Selection by Theory
   :header-rows: 1
   :widths: 25 30 45

   * - Target Function Type
     - Recommended Basis
     - Theoretical Justification
   * - Analytic on :math:`[-1,1]`
     - Chebyshev
     - Exponential convergence
   * - :math:`C^k` smooth
     - Legendre, Jacobi
     - :math:`O(1/n^k)` convergence
   * - Gaussian-weighted
     - Hermite
     - Optimal for :math:`e^{-x^2}` weights
   * - Exponential decay
     - Laguerre
     - Optimal for :math:`e^{-x}` weights
   * - Localized features
     - Gaussian RBF
     - Local approximation
   * - Multi-scale
     - Wavelets
     - Adaptive resolution
   * - Unknown/general
     - Chebyshev or GaussianRBF
     - Robust defaults

See :doc:`../basis_selection_guide` for practical guidance.
