# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Constraint utilities for KAN parameter transformations.

This module provides parameter transformations and Keras-compatible constraints
for enforcing mathematical requirements on trainable parameters.

Key Design Principle
--------------------
Instead of using hard constraints (like ``tf.maximum``) which have zero gradients
at boundaries, we use **smooth transformations** that:

1. Map unconstrained logits to constrained domains
2. Maintain smooth gradient flow everywhere
3. Allow optimizers to work in unbounded parameter space

Available Functions
-------------------
**Transformation Functions** (for use in forward pass):

- :func:`softplus_lower_bound` — Map to :math:`(L + \epsilon, \infty)`
- :func:`softplus_positive` — Map to :math:`(\epsilon, \infty)`
- :func:`sigmoid_interval` — Map to :math:`(L + \epsilon, H - \epsilon)`

**Inverse Transformations** (for initialization):

- :func:`inverse_softplus` — Invert softplus for initialization
- :func:`inverse_softplus_lower_bound` — Invert softplus_lower_bound

**Keras Constraints** (Keras-compatible constraint classes):

- :class:`PositivityConstraint` — Ensure strictly positive weights
- :class:`BoundedConstraint` — Ensure weights in bounded interval
- :class:`MonotonicityConstraint` — Ensure monotonic weight sequences
- :class:`OrthogonalityConstraint` — Soft orthogonality via regularization

Examples
--------
Using transformation functions in a custom layer:

>>> from arnold.layers.constraints import softplus_lower_bound, inverse_softplus_lower_bound
>>> 
>>> class MyLayer(tf.keras.layers.Layer):
...     def build(self, input_shape):
...         # Initialize logits to produce alpha=1.0 (for alpha > -0.5)
...         init_logits = inverse_softplus_lower_bound(
...             tf.constant(1.0), lower_bound=-0.5
...         )
...         self.alpha_logits = self.add_weight(
...             name="alpha_logits",
...             initializer=tf.keras.initializers.Constant(float(init_logits))
...         )
...     
...     def call(self, inputs):
...         # Transform logits to constrained alpha in forward pass
...         alpha = softplus_lower_bound(self.alpha_logits, lower_bound=-0.5)
...         # Use alpha in computation...

Using Keras constraints:

>>> from arnold.layers.constraints import PositivityConstraint
>>> layer = tf.keras.layers.Dense(
...     units=32,
...     kernel_constraint=PositivityConstraint()
... )

See Also
--------
- :mod:`arnold.layers.regularizers` for regularization utilities
- :mod:`arnold.layers.initializers` for weight initialization
"""

from arnold.layers.constraints.base import (
    KANConstraint,
)
from arnold.layers.constraints.bounds import (
    BoundedConstraint,
    inverse_softplus,
    inverse_softplus_lower_bound,
    sigmoid_interval,
    softplus_lower_bound,
    softplus_positive,
)
from arnold.layers.constraints.monotonicity import (
    MonotonicityConstraint,
)
from arnold.layers.constraints.orthogonality import (
    OrthogonalityConstraint,
)
from arnold.layers.constraints.positivity import (
    PositivityConstraint,
)

__all__ = [
    # Base
    "KANConstraint",
    # Transformation functions
    "softplus_lower_bound",
    "softplus_positive",
    "sigmoid_interval",
    "inverse_softplus",
    "inverse_softplus_lower_bound",
    # Keras constraints
    "PositivityConstraint",
    "BoundedConstraint",
    "MonotonicityConstraint",
    "OrthogonalityConstraint",
]
