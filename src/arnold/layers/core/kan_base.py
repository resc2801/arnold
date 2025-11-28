## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
import warnings
from abc import ABC
from typing import Literal

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl


def detect_hardware() -> Literal["cpu", "gpu", "tpu", "mps"]:
    """Detect the primary compute hardware available.
    
    Returns:
        "gpu" if CUDA/ROCm GPU is available
        "tpu" if TPU is available
        "mps" if Apple Metal Performance Shaders is available
        "cpu" otherwise
    """
    # Check for TPU
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        if resolver:
            return "tpu"
    except (ValueError, tf.errors.NotFoundError):
        pass
    
    # Check for GPU (CUDA/ROCm)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Check if it's Apple MPS
        for gpu in gpus:
            if "metal" in gpu.name.lower() or "mps" in gpu.name.lower():
                return "mps"
        return "gpu"
    
    # Check for MPS on macOS (Apple Silicon)
    try:
        # TensorFlow-metal registers as GPU but we can also check device details
        import platform
        if platform.system() == "Darwin" and platform.processor() == "arm":
            # Apple Silicon Mac without tensorflow-metal
            return "mps"
    except Exception:
        pass
    
    return "cpu"


def get_recommended_dtype(degree: int, hardware: str | None = None) -> tf.DType:
    """Get recommended compute dtype based on hardware and polynomial degree.
    
    Args:
        degree: The polynomial degree.
        hardware: Override hardware detection ("cpu", "gpu", "tpu", "mps").
                  If None, auto-detects.
    
    Returns:
        Recommended TensorFlow dtype for computation.
        
    Note:
        - CPU: float64 for degree > 10 (better precision), float32 otherwise
        - GPU: float32 (optimized for throughput)
        - TPU: bfloat16 or float32 (TPU-optimized)
        - MPS: float32 (Apple Metal limitation)
    """
    if hardware is None:
        hardware = detect_hardware()
    
    if hardware == "cpu":
        # CPU can handle float64 efficiently and benefits from precision
        return tf.float64 if degree > 10 else tf.float32
    elif hardware == "tpu":
        # TPU is optimized for bfloat16 but float32 is safer for polynomials
        return tf.float32
    else:  # gpu or mps
        # GPU/MPS: float32 for throughput, mixed precision handles the rest
        return tf.float32


@tfk.utils.register_keras_serializable(package="arnold", name="KANBase")
class KANBase(tfkl.Layer, ABC):
    r"""
    Abstract base class for Kolmogorov-Arnold Network layers.
    
    Supports hardware-adaptive dtype selection and mixed-precision training
    through ``tf.keras.mixed_precision``.
    """

    def __init__(
        self,
        units: int | None = None,
        input_dim: int | None = None,
        output_dim: int | None = None,
        use_bias: bool = True,
        activation=None,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        input_clip: tuple[float, float] | None = None,
        tanh_x: bool | None = None,
        compute_dtype: tf.DType | str | None = None,
        **kwargs,
    ):
        r"""
        :param units: Output dimensionality. Alias ``output_dim`` is supported for backwards compatibility.
        :param input_dim: Optional known input feature dimension; if omitted, inferred at build time.
        :param use_bias: Whether to include a bias term.
        :param activation: Optional activation applied to the layer output.
        :param kernel_regularizer: Regularizer for the coefficient/kernel weights (e.g., ``tf.keras.regularizers.L2(0.01)``).
        :param bias_regularizer: Regularizer for the bias vector.
        :param activity_regularizer: Regularizer applied to the layer output.
        :param input_clip: Optional (min, max) tuple for clamping inputs before basis evaluation.
        :param tanh_x: Deprecated. If True, applies ``tanh`` to inputs; prefer ``input_clip`` or explicit preprocessing.
        :param compute_dtype: Override dtype for basis computation. If None, respects the global
            mixed-precision policy (``tf.keras.mixed_precision.global_policy()``). Can be a
            ``tf.DType`` or string like ``"float32"``, ``"float64"``, ``"mixed_float16"``.
        """
        # Handle activity_regularizer through Keras base
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)

        resolved_units = units if units is not None else output_dim
        if resolved_units is None:
            raise ValueError("`units` (or legacy `output_dim`) must be provided.")

        # Emit deprecation warning if output_dim was used instead of units
        if output_dim is not None and units is None:
            warnings.warn(
                "The `output_dim` parameter is deprecated and will be removed in a future release. "
                "Please use `units` instead.",
                DeprecationWarning,
                stacklevel=3,
            )

        self.units = int(resolved_units)
        self.output_dim = self.units  # legacy alias
        self.input_dim = input_dim
        self.use_bias = use_bias
        self.activation = tfk.activations.get(activation)
        self.kernel_regularizer = tfk.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tfk.regularizers.get(bias_regularizer)
        self.input_clip = input_clip
        self.tanh_x = bool(tanh_x) if tanh_x is not None else False
        
        # Store user-specified compute dtype; None means use global policy
        self._user_compute_dtype = compute_dtype

        self.bias = None
        self.input_spec = None

    def build(self, input_shape):
        if len(input_shape) < 2:
            raise ValueError("KAN layers expect inputs with a feature dimension.")

        self.input_dim = int(input_shape[-1])
        self.input_spec = tfkl.InputSpec(ndim=len(input_shape), axes={-1: self.input_dim})

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer="zeros",
                regularizer=self.bias_regularizer,
                trainable=True,
            )
        super().build(input_shape)

    @property
    def effective_compute_dtype(self) -> tf.DType:
        """Get the effective compute dtype, respecting mixed-precision policy.
        
        Resolution order:
        1. User-specified compute_dtype (if provided)
        2. Global mixed-precision policy compute dtype
        3. Layer's variable dtype (fallback)
        """
        if self._user_compute_dtype is not None:
            if isinstance(self._user_compute_dtype, str):
                return tf.dtypes.as_dtype(self._user_compute_dtype)
            return self._user_compute_dtype
        
        # Respect Keras mixed-precision policy
        policy = tfk.mixed_precision.global_policy()
        return policy.compute_dtype or self.dtype
    
    def _cast_for_compute(self, tensor: tf.Tensor) -> tf.Tensor:
        """Cast tensor to effective compute dtype if different from current dtype."""
        target_dtype = self.effective_compute_dtype
        if tensor.dtype != target_dtype:
            return tf.cast(tensor, target_dtype)
        return tensor

    def _preprocess_inputs(self, inputs: tf.Tensor) -> tf.Tensor:
        x = tf.convert_to_tensor(inputs)
        if self.input_clip is not None:
            lo, hi = self.input_clip
            x = tf.clip_by_value(x, lo, hi)
        if self.tanh_x:
            x = tf.tanh(x)
        return x

    def _apply_activation_and_bias(self, outputs: tf.Tensor) -> tf.Tensor:
        y = outputs
        if self.use_bias and self.bias is not None:
            y = y + self.bias
        if self.activation is not None:
            y = self.activation(y)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)

    def get_config(self):
        base_config = super().get_config()
        
        # Serialize compute_dtype properly
        compute_dtype_str = None
        if self._user_compute_dtype is not None:
            if isinstance(self._user_compute_dtype, tf.DType):
                compute_dtype_str = self._user_compute_dtype.name
            else:
                compute_dtype_str = str(self._user_compute_dtype)
        
        config = {
            "units": self.units,
            "input_dim": self.input_dim,
            "use_bias": self.use_bias,
            "activation": tfk.activations.serialize(self.activation),
            "kernel_regularizer": tfk.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": tfk.regularizers.serialize(self.bias_regularizer),
            "input_clip": self.input_clip,
            "tanh_x": self.tanh_x,
            "output_dim": self.units,
            "compute_dtype": compute_dtype_str,
        }
        return {**base_config, **config}
