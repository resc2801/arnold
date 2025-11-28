.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _theory-index:

====================================================
Part I: Mathematical Foundations
====================================================

This part develops the complete mathematical theory underlying Kolmogorov-Arnold 
Networks. We begin with the foundational Kolmogorov-Arnold Representation Theorem 
and progressively build the theoretical framework needed to understand modern KAN 
architectures.

.. toctree::
   :maxdepth: 2
   :caption: Chapters

   kolmogorov_arnold_theorem
   kan_layers
   approximation_theory

----

Chapter Overview
----------------

:doc:`kolmogorov_arnold_theorem`
   The foundational theorem establishing that any continuous multivariate function 
   can be represented as superpositions of continuous univariate functions. We cover 
   the historical development, original theorem, proof structure, and constructive 
   extensions by Lorentz, Sprecher, and Köppen.

:doc:`kan_layers`
   Mathematical formalization of KAN layers. Covers the alternant tensor architecture,
   pseudo-Vandermonde matrices, polynomial/RBF/wavelet basis constructions, and the
   Tucker decomposition for memory efficiency.

:doc:`approximation_theory`
   Approximation-theoretic foundations connecting KANs to classical results. Jackson
   and Bernstein theorems, spectral convergence, optimal polynomial bases, and error 
   analysis for each basis family.

----

Reading Guide
-------------

**For Researchers:**

Start with :doc:`kolmogorov_arnold_theorem` for the theoretical foundations,
then :doc:`approximation_theory` for convergence guarantees.

**For Practitioners:**

:doc:`kan_layers` provides the mathematical details of the implementation.
Skip proofs on first read; focus on the main definitions and layer architectures.

**For Mathematicians:**

All three chapters are relevant. The proofs use standard techniques from
approximation theory, functional analysis, and multilinear algebra.

----

Notation Reference
------------------

**General:**

.. list-table::
   :widths: 15 85
   :header-rows: 0

   * - :math:`n`
     - Input dimension
   * - :math:`m`
     - Output dimension (number of units)
   * - :math:`d`
     - Polynomial degree
   * - :math:`K`
     - Number of basis functions (RBF grids, wavelet scales)

**Tensors and Matrices:**

.. list-table::
   :widths: 15 85
   :header-rows: 0

   * - :math:`\mathbf{V}(x)`
     - Pseudo-Vandermonde matrix, shape :math:`(n, d+1)`
   * - :math:`\mathcal{C}`
     - Alternant tensor (coefficients), shape :math:`(n, m, d+1)`
   * - :math:`\mathcal{G}`
     - Tucker core tensor
   * - :math:`\mathbf{U}^{(k)}`
     - Tucker factor matrices

**Function Spaces:**

.. list-table::
   :widths: 15 85
   :header-rows: 0

   * - :math:`C([a,b])`
     - Continuous functions on :math:`[a,b]`
   * - :math:`L^2_w`
     - Square-integrable functions with weight :math:`w`
   * - :math:`\Pi_n`
     - Polynomials of degree :math:`\leq n`
   * - :math:`E_n(f)`
     - Best approximation error by degree-:math:`n` polynomials

**Polynomials:**

.. list-table::
   :widths: 15 85
   :header-rows: 0

   * - :math:`T_n(x)`
     - Chebyshev polynomial of the first kind
   * - :math:`P_n(x)`
     - Legendre polynomial
   * - :math:`H_n(x)`
     - Hermite polynomial (physicist's)
   * - :math:`L_n^{(\alpha)}(x)`
     - Generalized Laguerre polynomial
   * - :math:`P_n^{(\alpha,\beta)}(x)`
     - Jacobi polynomial

----

Mathematical Prerequisites
--------------------------

This part assumes familiarity with:

- **Real analysis:** Continuity, compactness, uniform convergence
- **Linear algebra:** Matrix operations, tensor products, eigenvalues
- **Basic approximation theory:** Weierstrass theorem, best approximation

For readers new to approximation theory, we recommend:

1. Cheney, E.W. "Introduction to Approximation Theory"
2. DeVore, R.A. & Lorentz, G.G. "Constructive Approximation"
3. Trefethen, L.N. "Approximation Theory and Approximation Practice"
