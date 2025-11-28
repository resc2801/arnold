# Copyright (c) 2025 René Schubotz. All rights reserved.
# Type stubs for arnold

from typing import Literal

import tensorflow as tf

from arnold import layers as layers

__version__: str
__author__: str
__email__: str

def detect_hardware() -> Literal["cpu", "gpu", "tpu", "mps"]:
    """Detect the primary compute hardware available."""
    ...

def get_recommended_dtype(
    degree: int,
    hardware: str | None = None,
) -> tf.DType:
    """Get recommended compute dtype based on hardware and polynomial degree."""
    ...

__all__: list[str]
