# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""Sinc (Shannon sampling) basis KAN layer (stub)."""
import tensorflow as tf

from arnold.layers.core.spectral.base import SpectralBase

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold", name="SincBasis")
class SincBasis(SpectralBase):
    r"""
    Sinc basis KAN layer for band-limited interpolation.

    Uses sinc(x) = sin(πx)/(πx) basis functions (Shannon sampling theorem).

    .. warning::
        This is a stub implementation. Full implementation in Phase 9b.
    """

    def __init__(self, *, units: int, num_frequencies: int = 16, input_clip: tuple[float, float] | None = None, **kwargs):
        super().__init__(units=units, num_frequencies=num_frequencies, input_clip=input_clip, **kwargs)

    def _get_num_basis_functions(self) -> int:
        return self.num_frequencies

    def spectral_basis(self, x: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError("SincBasis is a stub. Full implementation in Phase 9b.")
