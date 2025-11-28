# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Recurrence-based polynomial sequences (Fibonacci-type and Lucas-type).

This module provides KAN layers based on polynomial sequences defined by
linear recurrence relations. These include:

**Fibonacci-type (multi-step recurrences):**

- :class:`Fibonacci` — Classical 2-term: :math:`F_n = x F_{n-1} + F_{n-2}`
- :class:`Tribonacci` — 3-term: :math:`T_n = x T_{n-1} + T_{n-2} + T_{n-3}`
- :class:`Tetranacci` — 4-term recurrence
- :class:`Pentanacci` — 5-term recurrence
- :class:`Hexanacci` — 6-term recurrence
- :class:`Heptanacci` — 7-term recurrence
- :class:`Octanacci` — 8-term recurrence

**Lucas polynomial sequences (w-polynomials):**

Defined by the general Lucas sequence:

.. math::

    w_0 = a, \quad w_1 = b, \quad w_{n+1} = p(x) w_n + q(x) w_{n-1}

- :class:`Lucas` — :math:`p(x) = x, q(x) = 1, (a, b) = (2, x)`
- :class:`Pell` — :math:`p(x) = 2x, q(x) = 1, (a, b) = (0, 1)`
- :class:`PellLucas` — :math:`p(x) = 2x, q(x) = 1, (a, b) = (2, 2x)`
- :class:`Fermat` — :math:`p(x) = 3x, q(x) = -2, (a, b) = (0, 1)`
- :class:`FermatLucas` — :math:`p(x) = 3x, q(x) = -2, (a, b) = (2, 3x)`
- :class:`Jacobsthal` — :math:`p(x) = 1, q(x) = 2x, (a, b) = (0, 1)`
- :class:`JacobsthalLucas` — :math:`p(x) = 1, q(x) = 2x, (a, b) = (2, 1)`

Mathematical Background
-----------------------
These sequences generalize integer sequences to polynomial families. The
recurrence structure makes them efficient to evaluate via iterative schemes.

For Fibonacci-type, the growth is polynomial in :math:`x` with degree
approximately :math:`n`. For Lucas-type, growth depends on :math:`p(x)` and
can be exponential; use ``input_clip`` for stability.

References
----------
.. [1] Horadam, A.F. "Basic properties of a certain generalized sequence
       of numbers." Fibonacci Quarterly, 1965.
.. [2] https://mathworld.wolfram.com/LucasPolynomialSequence.html
.. [3] https://en.wikipedia.org/wiki/Fibonacci_polynomials
"""

# Fibonacci-type (multi-step recurrences)
from .fermat import Fermat, FermatLucas
from .fibonacci import Fibonacci
from .heptanacci import Heptanacci
from .hexanacci import Hexanacci
from .jacobsthal import Jacobsthal, JacobsthalLucas

# Lucas polynomial sequences (w-polynomials)
from .lucas import Lucas
from .octanacci import Octanacci
from .pell import Pell, PellLucas
from .pentanacci import Pentanacci
from .tetranacci import Tetranacci
from .tribonacci import Tribonacci


__all__ = [
    # Fibonacci-type
    "Fibonacci",
    "Tribonacci",
    "Tetranacci",
    "Pentanacci",
    "Hexanacci",
    "Heptanacci",
    "Octanacci",
    # Lucas-type
    "Lucas",
    "Pell",
    "PellLucas",
    "Fermat",
    "FermatLucas",
    "Jacobsthal",
    "JacobsthalLucas",
]
