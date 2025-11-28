# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Common utilities for ARNOLD KAN layers.

This module provides shared infrastructure for all basis function implementations:

- **types**: Type definitions, protocols, and enums
- **evaluation**: Recurrence evaluation strategies (Clenshaw, Horner)
- **parameters**: Domain scaling and parameter handling
- **numerics**: Numerical utilities for stability

Submodules
----------
types
    Type definitions, protocols, and domain specifications.
evaluation
    Polynomial evaluation algorithms (Clenshaw, Horner, recurrence).
parameters
    Domain transformations and parameter validation.
numerics
    Numerically stable computation primitives.

Import DAG
----------
All modules in ``core/`` may import from ``common/``.
``common/`` MUST NOT import from any sibling modules in ``core/``.

Example
-------
>>> from arnold.layers.core.common import DomainSpec, clenshaw_eval
>>> from arnold.layers.core.common.numerics import safe_log
"""

from arnold.layers.core.common.types import (
    DomainSpec,
    BasisProtocol,
    NormalizationScheme,
    EvaluationStrategy,
    RecurrenceCoefficients,
    DOMAIN_UNIT_INTERVAL,
    DOMAIN_SYMMETRIC,
    DOMAIN_POSITIVE,
    DOMAIN_REAL_LINE,
    DOMAIN_UNIT_CIRCLE,
)
from arnold.layers.core.common.evaluation import (
    clenshaw_eval,
    horner_eval,
    three_term_recurrence,
)
from arnold.layers.core.common.parameters import (
    scale_to_domain,
    get_domain_bounds,
    validate_parameters,
    clip_to_domain,
    normalize_input,
)
from arnold.layers.core.common.numerics import (
    safe_log,
    safe_sqrt,
    safe_divide,
    kahan_sum,
    log_pochhammer,
    log_factorial,
    log_binomial,
    stabilize_recurrence,
)

__all__ = [
    # Types
    "DomainSpec",
    "BasisProtocol",
    "NormalizationScheme",
    "EvaluationStrategy",
    "RecurrenceCoefficients",
    # Domain constants
    "DOMAIN_UNIT_INTERVAL",
    "DOMAIN_SYMMETRIC",
    "DOMAIN_POSITIVE",
    "DOMAIN_REAL_LINE",
    "DOMAIN_UNIT_CIRCLE",
    # Evaluation
    "clenshaw_eval",
    "horner_eval",
    "three_term_recurrence",
    # Parameters
    "scale_to_domain",
    "get_domain_bounds",
    "validate_parameters",
    "clip_to_domain",
    "normalize_input",
    # Numerics
    "safe_log",
    "safe_sqrt",
    "safe_divide",
    "kahan_sum",
    "log_pochhammer",
    "log_factorial",
    "log_binomial",
    "stabilize_recurrence",
]
