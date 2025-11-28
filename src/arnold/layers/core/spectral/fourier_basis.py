# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""Fourier/trigonometric basis KAN layer (stub)."""
import tensorflow as tf

from arnold.layers.core.spectral.base import SpectralBase

tfk = tf.keras


@tfk.utils.register_keras_serializable(package="arnold", name="FourierKAN")
class FourierKAN(SpectralBase):
    r"""
    Fourier/trigonometric basis KAN layer.

    Basis: {1, cos(ωx), sin(ωx), cos(2ωx), sin(2ωx), ..., cos(nωx), sin(nωx)}
    Number of basis functions: 2 * degree + 1

    .. warning::
        This is a stub implementation. Full implementation in Phase 9b.
    """

    def __init__(
        self,
        *,
        units: int,
        degree: int = 8,
        frequency: float = 1.0,
        learnable_frequency: bool = False,
        input_clip: tuple[float, float] | None = None,
        **kwargs,
    ):
        self.degree = degree
        self.frequency_init = frequency
        self.learnable_frequency = learnable_frequency
        super().__init__(units=units, num_frequencies=degree, input_clip=input_clip, **kwargs)

    def _get_num_basis_functions(self) -> int:
        return 2 * self.degree + 1

    def spectral_basis(self, x: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError("FourierKAN is a stub. Full implementation in Phase 9b.")

    def get_config(self):
        config = super().get_config()
        config.update({
            "degree": self.degree,
            "frequency": self.frequency_init,
            "learnable_frequency": self.learnable_frequency,
        })
        return config
