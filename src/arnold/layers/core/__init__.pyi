# Copyright (c) 2025 René Schubotz. All rights reserved.
# Type stubs for arnold.layers.core

from abc import ABC
from collections.abc import Callable
from typing import Any, Literal

import tensorflow as tf
from tensorflow import keras

# Utility functions
def detect_hardware() -> Literal["cpu", "gpu", "tpu", "mps"]: ...
def get_recommended_dtype(degree: int, hardware: str | None = None) -> tf.DType: ...

# Base Classes
class KANBase(keras.layers.Layer, ABC):
    """Abstract base class for Kolmogorov-Arnold Network layers."""

    units: int
    output_dim: int  # Legacy alias for units
    input_dim: int | None
    use_bias: bool
    activation: Callable[[tf.Tensor], tf.Tensor] | None
    kernel_regularizer: keras.regularizers.Regularizer | None
    bias_regularizer: keras.regularizers.Regularizer | None
    input_clip: tuple[float, float] | None
    tanh_x: bool
    bias: tf.Variable | None

    def __init__(
        self,
        units: int | None = None,
        input_dim: int | None = None,
        output_dim: int | None = None,  # Deprecated
        use_bias: bool = True,
        activation: str | Callable[[tf.Tensor], tf.Tensor] | None = None,
        kernel_regularizer: keras.regularizers.Regularizer | str | dict[str, Any] | None = None,
        bias_regularizer: keras.regularizers.Regularizer | str | dict[str, Any] | None = None,
        activity_regularizer: keras.regularizers.Regularizer | str | dict[str, Any] | None = None,
        input_clip: tuple[float, float] | None = None,
        tanh_x: bool | None = None,
        compute_dtype: tf.DType | str | None = None,
        **kwargs: Any,
    ) -> None: ...

    @property
    def effective_compute_dtype(self) -> tf.DType: ...

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor: ...
    def get_config(self) -> dict[str, Any]: ...


class PolynomialBase(KANBase):
    """Abstract base class for polynomial KAN layers."""

    degree: int
    core_ranks: tuple[int, int, int] | None
    promote_to_float64: bool
    precision_threshold: int
    use_clenshaw: bool | None
    hardware_adaptive: bool
    poly_coeffs: tf.Variable | None
    poly_coeffs_core: tf.Variable | None
    poly_coeffs_A: tf.Variable | None
    poly_coeffs_B: tf.Variable | None
    poly_coeffs_C: tf.Variable | None

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        core_ranks: tuple[int, int, int] | None = None,
        input_clip: tuple[float, float] | None = None,
        promote_to_float64: bool | None = None,
        precision_threshold: int = 10,
        use_clenshaw: bool | None = None,
        hardware_adaptive: bool = True,
        **kwargs: Any,
    ) -> None: ...

    def pseudo_vandermonde(self, x: tf.Tensor) -> tf.Tensor: ...
    def clenshaw_basis(self, x: tf.Tensor) -> tf.Tensor: ...


class RBFBase(KANBase):
    """Abstract base class for RBF KAN layers."""

    num_grids: int
    grid_min: float
    grid_max: float
    grid_eps: float
    kernel_weights: tf.Variable | None
    grid_tensor: tf.Variable | None

    def __init__(
        self,
        num_grids: int,
        *,
        units: int,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        grid_eps: float = 0.02,
        **kwargs: Any,
    ) -> None: ...

    def rbf_basis(self, x: tf.Tensor) -> tf.Tensor: ...


class WaveletBase(KANBase):
    """Abstract base class for wavelet KAN layers."""

    num_grids: int
    grid_min: float
    grid_max: float
    wavelet_weights: tf.Variable | None
    scale_logits: tf.Variable | None
    translation: tf.Variable | None

    def __init__(
        self,
        num_grids: int,
        *,
        units: int,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        **kwargs: Any,
    ) -> None: ...

    def wavelet_basis(self, x: tf.Tensor) -> tf.Tensor: ...


# Polynomial Layers - Orthogonal
class Legendre(PolynomialBase):
    orthonormal: bool
    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        orthonormal: bool = False,
        **kwargs: Any,
    ) -> None: ...

class Chebyshev1st(PolynomialBase):
    orthonormal: bool
    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        orthonormal: bool = False,
        **kwargs: Any,
    ) -> None: ...

class Chebyshev2nd(PolynomialBase):
    orthonormal: bool
    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        orthonormal: bool = False,
        **kwargs: Any,
    ) -> None: ...

class Chebyshev3rd(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        **kwargs: Any,
    ) -> None: ...

class Chebyshev4th(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        **kwargs: Any,
    ) -> None: ...

class Jacobi(PolynomialBase):
    alpha: float
    beta: float
    alpha_trainable: bool
    beta_trainable: bool
    def __init__(
        self,
        degree: int,
        *,
        units: int,
        alpha_init: float | None = None,
        alpha_trainable: bool = True,
        beta_init: float | None = None,
        beta_trainable: bool = True,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        **kwargs: Any,
    ) -> None: ...

class Gegenbauer(PolynomialBase):
    alpha_init: float | None
    alpha_trainable: bool
    def __init__(
        self,
        degree: int,
        *,
        units: int,
        alpha_init: float | None = None,
        alpha_trainable: bool = True,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        orthonormal: bool = False,
        **kwargs: Any,
    ) -> None: ...

class Hermite(PolynomialBase):
    normalized: bool
    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = (-5.0, 5.0),
        normalized: bool = False,
        **kwargs: Any,
    ) -> None: ...

class GeneralizedLaguerre(PolynomialBase):
    alpha_init: float | None
    alpha_trainable: bool
    def __init__(
        self,
        degree: int,
        *,
        units: int,
        alpha_init: float | None = None,
        alpha_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class Bessel(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class Charlier(PolynomialBase):
    a_init: float | None
    a_trainable: bool
    def __init__(
        self,
        degree: int,
        *,
        units: int,
        a_init: float | None = None,
        a_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class Wilson(PolynomialBase):
    orthonormal: bool
    def __init__(
        self,
        degree: int,
        *,
        units: int,
        a_init: float | None = None,
        a_trainable: bool = True,
        b_init: float | None = None,
        b_trainable: bool = True,
        c_init: float | None = None,
        c_trainable: bool = True,
        d_init: float | None = None,
        d_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        orthonormal: bool = False,
        **kwargs: Any,
    ) -> None: ...

class Pollaczek(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int,
        lambda_init: float | None = None,
        lambda_trainable: bool = True,
        phi_init: float | None = None,
        phi_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class AssociatedMeixnerPollaczek(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int,
        lambda_init: float | None = None,
        lambda_trainable: bool = True,
        phi_init: float | None = None,
        phi_trainable: bool = True,
        c_init: float | None = None,
        c_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class BannaiIto(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int,
        a_init: float | None = None,
        a_trainable: bool = True,
        b_init: float | None = None,
        b_trainable: bool = True,
        c_init: float | None = None,
        c_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class AlSalamCarlitz1st(PolynomialBase):
    a_init: float | None
    a_trainable: bool
    q_init: float | None
    q_trainable: bool
    def __init__(
        self,
        degree: int,
        *,
        units: int,
        a_init: float | None = None,
        a_trainable: bool = True,
        q_init: float | None = None,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class AlSalamCarlitz2nd(PolynomialBase):
    a_init: float | None
    a_trainable: bool
    q_init: float | None
    q_trainable: bool
    def __init__(
        self,
        degree: int,
        *,
        units: int,
        a_init: float | None = None,
        a_trainable: bool = True,
        q_init: float | None = None,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class AskeyWilson(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int,
        a_init: float | None = None,
        a_trainable: bool = True,
        b_init: float | None = None,
        b_trainable: bool = True,
        c_init: float | None = None,
        c_trainable: bool = True,
        d_init: float | None = None,
        d_trainable: bool = True,
        q_init: float | None = None,
        q_trainable: bool = True,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...


# Polynomial Layers - Discrete Orthogonal
class Krawtchouk(PolynomialBase):
    """Krawtchouk polynomial KAN layer K_n(x; p, N).

    Orthogonal on {0, 1, ..., N} with binomial weight.
    """
    def __init__(
        self,
        degree: int,
        *,
        units: int,
        p: float = 0.5,
        N: int = 10,
        trainable_params: bool = False,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class Hahn(PolynomialBase):
    """Hahn polynomial KAN layer Q_n(x; alpha, beta, N).

    Generalization of Krawtchouk with two shape parameters.
    Orthogonal on {0, 1, ..., N} with hypergeometric weight.
    """
    def __init__(
        self,
        degree: int,
        *,
        units: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        N: int = 10,
        trainable_params: bool = False,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class Meixner(PolynomialBase):
    """Meixner polynomial KAN layer M_n(x; beta, c).

    Orthogonal on {0, 1, 2, ...} with negative binomial weight.
    """
    def __init__(
        self,
        degree: int,
        *,
        units: int,
        beta: float = 2.0,
        c: float = 0.5,
        trainable_params: bool = False,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class Racah(PolynomialBase):
    """Racah polynomial KAN layer R_n(lambda(x); alpha, beta, gamma, delta).

    The most general classical discrete orthogonal polynomials.
    lambda(x) = x(x + gamma + delta + 1)
    """
    def __init__(
        self,
        degree: int,
        *,
        units: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.5,
        delta: float = 0.5,
        N: int = 10,
        trainable_params: bool = False,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...


# Polynomial Layers - Fibonacci-type
class Fibonacci(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class Lucas(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class Pell(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class PellLucas(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class Fermat(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class FermatLucas(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class Jacobsthal(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class JacobsthalLucas(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class Tetranacci(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class Pentanacci(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class Hexanacci(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class Heptanacci(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class Octanacci(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...


# Polynomial Layers - Other
class Boubaker(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...

class Laurent(PolynomialBase):
    def __init__(
        self,
        degree: int,
        *,
        units: int | None = None,
        output_dim: int | None = None,
        input_clip: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None: ...


# RBF Layers
class GaussianRBF(RBFBase):
    epsilon_init: float | None
    epsilon_trainable: bool
    def __init__(
        self,
        num_grids: int,
        *,
        units: int,
        epsilon_init: float | None = None,
        epsilon_trainable: bool = True,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        grid_eps: float = 0.02,
        **kwargs: Any,
    ) -> None: ...

class ExponentialRBF(RBFBase):
    sigma_init: float | None
    sigma_trainable: bool
    def __init__(
        self,
        num_grids: int,
        *,
        units: int,
        sigma_init: float | None = None,
        sigma_trainable: bool = True,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        grid_eps: float = 0.02,
        **kwargs: Any,
    ) -> None: ...

class MultiquadricRBF(RBFBase):
    epsilon_init: float | None
    epsilon_trainable: bool
    def __init__(
        self,
        num_grids: int,
        *,
        units: int,
        epsilon_init: float | None = None,
        epsilon_trainable: bool = True,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        grid_eps: float = 0.02,
        **kwargs: Any,
    ) -> None: ...

class InverseMultiQuadricRBF(RBFBase):
    epsilon_init: float | None
    epsilon_trainable: bool
    def __init__(
        self,
        num_grids: int,
        *,
        units: int,
        epsilon_init: float | None = None,
        epsilon_trainable: bool = True,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        grid_eps: float = 0.02,
        **kwargs: Any,
    ) -> None: ...

class InverseQuadricRBF(RBFBase):
    epsilon_init: float | None
    epsilon_trainable: bool
    def __init__(
        self,
        num_grids: int,
        *,
        units: int,
        epsilon_init: float | None = None,
        epsilon_trainable: bool = True,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        grid_eps: float = 0.02,
        **kwargs: Any,
    ) -> None: ...

class CauchyRBF(RBFBase):
    gamma_init: float | None
    gamma_trainable: bool
    def __init__(
        self,
        num_grids: int,
        *,
        units: int,
        gamma_init: float | None = None,
        gamma_trainable: bool = True,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        grid_eps: float = 0.02,
        **kwargs: Any,
    ) -> None: ...

class CubicRBF(RBFBase):
    def __init__(
        self,
        num_grids: int,
        *,
        units: int,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        grid_eps: float = 0.02,
        **kwargs: Any,
    ) -> None: ...

class LinearRBF(RBFBase):
    def __init__(
        self,
        num_grids: int,
        *,
        units: int,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        grid_eps: float = 0.02,
        **kwargs: Any,
    ) -> None: ...

class PowerRBF(RBFBase):
    power: float
    def __init__(
        self,
        num_grids: int,
        *,
        units: int,
        power: float = 2.0,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        grid_eps: float = 0.02,
        **kwargs: Any,
    ) -> None: ...

class ThinPlateSplineRBF(RBFBase):
    def __init__(
        self,
        num_grids: int,
        *,
        units: int,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        grid_eps: float = 0.02,
        **kwargs: Any,
    ) -> None: ...


# Wavelet Layers
class Ricker(WaveletBase):
    def __init__(
        self,
        num_grids: int,
        *,
        units: int,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        **kwargs: Any,
    ) -> None: ...

class Morelet(WaveletBase):
    omega0: float
    def __init__(
        self,
        num_grids: int,
        *,
        units: int,
        omega0: float = 5.0,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        **kwargs: Any,
    ) -> None: ...

class DerivativeOfGaussian(WaveletBase):
    order: int
    def __init__(
        self,
        num_grids: int,
        *,
        units: int,
        order: int = 1,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        **kwargs: Any,
    ) -> None: ...

class Meyer(WaveletBase):
    def __init__(
        self,
        num_grids: int,
        *,
        units: int,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        **kwargs: Any,
    ) -> None: ...

class Shannon(WaveletBase):
    def __init__(
        self,
        num_grids: int,
        *,
        units: int,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        **kwargs: Any,
    ) -> None: ...

class Bump(WaveletBase):
    def __init__(
        self,
        num_grids: int,
        *,
        units: int,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        **kwargs: Any,
    ) -> None: ...

class Poisson(WaveletBase):
    n: int
    def __init__(
        self,
        num_grids: int,
        *,
        units: int,
        n: int = 1,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        **kwargs: Any,
    ) -> None: ...


__all__: list[str]
