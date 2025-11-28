.. Copyright (c) 2025 René Schubotz. All rights reserved.

.. _symbolic-tools:

====================================================
Symbolic Tools
====================================================

Convert trained KAN layers to symbolic mathematical expressions.

Overview
--------

The ``symbolic`` module provides tools for **interpretability** — extracting
explicit mathematical formulas from trained KAN layers. This is one of the
key advantages of KANs over traditional neural networks.

**Key Features:**

- Convert trained layers to SymPy expressions
- Export to LaTeX for publications
- Extract and manipulate learned coefficients
- Simplify and analyze expressions

Quick Start
-----------

.. code-block:: python

   from arnold.layers import Legendre
   from arnold.layers.symbolic import kan_to_polynomial, kan_to_latex
   
   # Create and train a layer
   layer = Legendre(units=1, degree=3)
   layer.build((None, 2))
   # ... train the layer ...
   
   # Convert to symbolic expression
   expr = kan_to_polynomial(layer)
   print(expr)  # e.g., 0.5*x_0**2 - 0.3*x_1 + 1.2*x_0*x_1
   
   # Get LaTeX for papers
   latex = kan_to_latex(layer, mode='equation')
   print(latex)
   # \begin{equation}
   # f(x) = 0.5 x_{0}^{2} - 0.3 x_{1} + 1.2 x_{0} x_{1}
   # \end{equation}


API Reference
-------------

kan_to_polynomial
~~~~~~~~~~~~~~~~~

Convert a trained KAN layer to a SymPy expression.

.. code-block:: python

   from arnold.layers.symbolic import kan_to_polynomial
   
   expr = kan_to_polynomial(
       layer,                    # Trained KAN layer
       input_symbols=None,       # Custom symbols [x_0, x_1, ...]
       simplify=True,           # Simplify the result
   )

**Parameters:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``layer``
     - Required
     - Trained KAN layer (KANBase subclass)
   * - ``input_symbols``
     - None
     - Custom SymPy symbols; default: ``[x_0, x_1, ...]``
   * - ``simplify``
     - True
     - Apply SymPy simplification

**Returns:** ``sympy.Expr``


kan_to_latex
~~~~~~~~~~~~

Convert a trained KAN layer to LaTeX.

.. code-block:: python

   from arnold.layers.symbolic import kan_to_latex
   
   latex = kan_to_latex(
       layer,
       simplify=True,
       mode='inline',    # 'inline', 'equation', or 'align'
   )

**Modes:**

- ``'inline'``: Just the expression
- ``'equation'``: Wrapped in ``\\begin{equation}``
- ``'align'``: Wrapped in ``\\begin{align}``


extract_coefficients
~~~~~~~~~~~~~~~~~~~~

Get raw coefficient values from a layer.

.. code-block:: python

   from arnold.layers.symbolic import extract_coefficients
   
   coeffs = extract_coefficients(layer, as_dict=True)
   print(coeffs.keys())  # ['legendre/kernel:0', 'legendre/bias:0']
   print(coeffs['legendre/kernel:0'].shape)  # (2, 1, 4)


Simplification Utilities
------------------------

The ``symbolic`` module also provides expression manipulation tools:

.. code-block:: python

   from arnold.layers.symbolic import (
       simplify_expression,
       expand_expression,
       factor_expression,
       collect_terms,
       trigsimp_expression,
       polynomial_degree,
       coefficient_list,
   )
   
   import sympy as sp
   x = sp.Symbol('x')
   
   # Simplify
   expr = x**2 + 2*x + 1
   simplified = simplify_expression(expr)  # (x + 1)**2
   
   # Expand
   expr = (x + 1)**3
   expanded = expand_expression(expr)  # x**3 + 3*x**2 + 3*x + 1
   
   # Factor
   expr = x**2 - 1
   factored = factor_expression(expr)  # (x - 1)*(x + 1)
   
   # Collect terms
   expr = x*y + x - 3 + 2*x**2 - y*x**2
   collected = collect_terms(expr, x)  # x**2*(2 - y) + x*(y + 1) - 3
   
   # Polynomial degree and coefficients
   expr = 3*x**4 + 2*x**2 - 1
   deg = polynomial_degree(expr, x)  # 4
   coeffs = coefficient_list(expr, x)  # [3, 0, 2, 0, -1]


Supported Layer Types
---------------------

The symbolic backend currently supports:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Layer Type
     - Symbolic Representation
   * - Legendre
     - Legendre polynomial expansion
   * - Chebyshev (1st, 2nd)
     - Chebyshev polynomial expansion
   * - Hermite
     - Hermite polynomial expansion
   * - FourierKAN
     - Trigonometric series (sin/cos)
   * - Generic polynomial
     - Power series fallback

For unsupported layers, a generic power series representation is used.


Requirements
------------

The symbolic module requires **SymPy**:

.. code-block:: bash

   pip install sympy

If SymPy is not installed, a helpful error message is shown when
attempting to use symbolic functions.


Use Cases
---------

**1. Scientific Publications**

.. code-block:: python

   # Get a formula you can put directly in your paper
   latex = kan_to_latex(trained_layer, mode='equation')

**2. Symbolic Differentiation**

.. code-block:: python

   import sympy as sp
   expr = kan_to_polynomial(layer)
   x = sp.Symbol('x_0')
   derivative = sp.diff(expr, x)

**3. Function Analysis**

.. code-block:: python

   expr = kan_to_polynomial(layer)
   # Find critical points
   x = sp.Symbol('x_0')
   critical = sp.solve(sp.diff(expr, x), x)

**4. Export to Other Systems**

.. code-block:: python

   # Export to Mathematica
   expr = kan_to_polynomial(layer)
   mathematica_code = sp.mathematica_code(expr)
   
   # Export to MATLAB
   matlab_code = sp.octave_code(expr)

