# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Orthogonality constraint for KAN weights.

This module provides constraints ensuring weight matrices have orthogonal rows
or columns.
"""

from __future__ import annotations

import tensorflow as tf

from arnold.layers.constraints.base import KANConstraint

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold")
class OrthogonalityConstraint(KANConstraint):
    r"""
    Keras constraint enforcing orthonormality on weight matrices.

    Applies Gram-Schmidt orthonormalization or projects onto the
    Stiefel manifold to ensure rows or columns are orthonormal.

    Given weight matrix :math:`W`, projects to nearest orthonormal matrix
    :math:`\tilde{W}` such that:

    .. math::

        \tilde{W}^\top \tilde{W} = I \quad \text{(for column orthonormality)}

    or

    .. math::

        \tilde{W} \tilde{W}^\top = I \quad \text{(for row orthonormality)}

    Parameters
    ----------
    mode : {'rows', 'columns'}, default='rows'
        Whether to orthonormalize rows or columns.
    method : {'svd', 'gram_schmidt'}, default='svd'
        Orthonormalization method.
        - 'svd': Project via SVD (more stable).
        - 'gram_schmidt': Classical Gram-Schmidt (faster).

    Notes
    -----
    SVD projection computes:

    .. math::

        \tilde{W} = U V^\top

    where :math:`W = U \Sigma V^\top` is the SVD of :math:`W`.

    Examples
    --------
    >>> # For orthonormal rows (output dimensions):
    >>> constraint = OrthogonalityConstraint(mode='rows')
    >>>
    >>> # For orthonormal columns (input dimensions):
    >>> constraint = OrthogonalityConstraint(mode='columns')
    """

    def __init__(
        self, mode: str = "rows", method: str = "svd", **kwargs
    ):
        super().__init__(**kwargs)
        if mode not in ("rows", "columns"):
            raise ValueError(f"mode must be 'rows' or 'columns', got {mode}")
        if method not in ("svd", "gram_schmidt"):
            raise ValueError(f"method must be 'svd' or 'gram_schmidt', got {method}")
        self.mode = mode
        self.method = method

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        """Apply orthonormality constraint."""
        # Handle 2D case
        if len(w.shape) != 2:
            # For higher-rank tensors, reshape to 2D
            original_shape = tf.shape(w)
            w = tf.reshape(w, [original_shape[0], -1])
            reshaped = True
        else:
            reshaped = False
            original_shape = None

        # Transpose if orthonormalizing columns
        if self.mode == "columns":
            w = tf.transpose(w)

        # Apply orthonormalization
        if self.method == "svd":
            w_ortho = self._svd_project(w)
        else:
            w_ortho = self._gram_schmidt(w)

        # Transpose back if needed
        if self.mode == "columns":
            w_ortho = tf.transpose(w_ortho)

        # Reshape back if needed
        if reshaped:
            w_ortho = tf.reshape(w_ortho, original_shape)

        return w_ortho

    def _svd_project(self, w: tf.Tensor) -> tf.Tensor:
        """Project onto Stiefel manifold via SVD."""
        s, u, v = tf.linalg.svd(w, full_matrices=False)
        del s  # Not needed
        return tf.matmul(u, v, adjoint_b=True)

    def _gram_schmidt(self, w: tf.Tensor) -> tf.Tensor:
        """Apply Gram-Schmidt orthonormalization."""
        # Work with rows
        rows = tf.unstack(w, axis=0)
        ortho_rows = []

        for row in rows:
            # Subtract projections onto previous orthonormal rows
            for ortho_row in ortho_rows:
                proj = tf.reduce_sum(row * ortho_row) * ortho_row
                row = row - proj

            # Normalize
            norm = tf.norm(row) + 1e-8
            row = row / norm
            ortho_rows.append(row)

        return tf.stack(ortho_rows, axis=0)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"mode": self.mode, "method": self.method})
        return config
