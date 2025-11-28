#!/usr/bin/env python
# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""
Performance benchmark for polynomial KAN layers.

Features:
- Timing comparison: pseudo_vandermonde vs clenshaw paths
- Memory profiling with peak memory tracking
- Hardware detection and recommendations
- TPU support (when available)

Usage:
    python benchmarks/poly_bench.py --degree 32 --repeats 50 --mode basis
    python benchmarks/poly_bench.py --degree 16 --mode memory
    python benchmarks/poly_bench.py --mode hardware-info
"""
import argparse
import time
import gc
import os
import sys
from typing import Callable, Any

import numpy as np
import tensorflow as tf

from arnold.layers.core.polynomial.orthogonal import (
    Chebyshev1st,
    Chebyshev2nd,
    Chebyshev3rd,
    Chebyshev4th,
    GeneralizedLaguerre,
    Gegenbauer,
    Hermite,
    Jacobi,
    Legendre,
    Bessel,
    BannaiIto,
    Charlier,
)
from arnold import detect_hardware, get_recommended_dtype


# =============================================================================
# Memory Profiling Utilities
# =============================================================================

def get_memory_info() -> dict[str, float]:
    """Get current memory usage in MB."""
    import resource
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "rss_mb": rusage.ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else rusage.ru_maxrss / 1024,
    }


def measure_memory_peak(fn: Callable[[], Any], warmup: int = 2) -> dict[str, float]:
    """Measure peak memory during function execution."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    # Clear memory and get baseline
    gc.collect()
    tf.keras.backend.clear_session()
    gc.collect()
    
    baseline = get_memory_info()
    
    # Run function
    result = fn()
    
    # Get peak
    peak = get_memory_info()
    
    return {
        "baseline_mb": baseline["rss_mb"],
        "peak_mb": peak["rss_mb"],
        "delta_mb": peak["rss_mb"] - baseline["rss_mb"],
    }


# =============================================================================
# Timing Utilities
# =============================================================================

def _timeit(fn: Callable[[], Any], repeats: int) -> float:
    """Time a function over multiple repeats, return average in seconds."""
    # Warmup
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / repeats


def benchmark_basis(layer, x: tf.Tensor, repeats: int, method: str) -> float:
    """Benchmark basis evaluation."""
    if method == "pseudo":
        fn = tf.function(lambda: layer.pseudo_vandermonde(x), jit_compile=True, reduce_retracing=True)
    else:
        fn = tf.function(lambda: layer.clenshaw_basis(x), jit_compile=True, reduce_retracing=True)
    return _timeit(fn, repeats)


def benchmark_forward(layer, x: tf.Tensor, repeats: int) -> float:
    """Benchmark full forward pass."""
    fn = tf.function(lambda: layer(x), jit_compile=True, reduce_retracing=True)
    return _timeit(fn, repeats)


# =============================================================================
# Hardware Detection
# =============================================================================

def print_hardware_info():
    """Print detailed hardware information."""
    print("=" * 60)
    print("HARDWARE DETECTION REPORT")
    print("=" * 60)
    
    # ARNOLD detection
    hw = detect_hardware()
    print(f"\nARNOLD detect_hardware(): {hw}")
    
    # TensorFlow devices
    print("\nTensorFlow Devices:")
    for device in tf.config.list_physical_devices():
        print(f"  - {device.device_type}: {device.name}")
    
    # GPU details if available
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\nGPU Count: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            try:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"  GPU {i}: {details}")
            except Exception:
                print(f"  GPU {i}: {gpu.name}")
    
    # TPU check
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        print(f"\nTPU Available: {resolver.cluster_spec()}")
    except (ValueError, tf.errors.NotFoundError):
        print("\nTPU Available: No")
    
    # Recommended dtypes
    print("\nRecommended dtypes by degree:")
    for degree in [5, 10, 15, 20, 30]:
        dtype = get_recommended_dtype(degree, hw)
        print(f"  degree={degree:2d}: {dtype.name}")
    
    # Mixed precision policy
    print(f"\nCurrent mixed-precision policy: {tf.keras.mixed_precision.global_policy().name}")
    print("=" * 60)


# =============================================================================
# Memory Benchmark Mode
# =============================================================================

def run_memory_benchmark(args):
    """Run memory benchmarks for different polynomial families."""
    print("=" * 60)
    print("MEMORY PROFILING BENCHMARK")
    print(f"degree={args.degree}, input_dim={args.input_dim}, batch={args.batch}")
    print("=" * 60)
    
    x = tf.random.uniform((args.batch, args.input_dim), minval=-1.0, maxval=1.0, dtype=tf.float32)
    
    families = [
        (Legendre, {"input_clip": (-1.0, 1.0)}),
        (Chebyshev1st, {"input_clip": (-1.0, 1.0)}),
        (Jacobi, {"input_clip": (-1.0, 1.0), "alpha_init": 0.5, "beta_init": -0.25}),
        (Hermite, {"input_clip": None}),
        (GeneralizedLaguerre, {"input_clip": (0.0, 4.0), "alpha_init": 0.3}),
    ]
    
    print(f"\n{'Family':<20} {'Baseline MB':>12} {'Peak MB':>12} {'Delta MB':>12}")
    print("-" * 60)
    
    for cls, extra in families:
        gc.collect()
        tf.keras.backend.clear_session()
        gc.collect()
        
        def test_fn():
            layer = cls(
                degree=args.degree, 
                units=4, 
                promote_to_float64=args.include_promote,
                use_clenshaw=False,  # Avoid Clenshaw bugs for now
                **extra
            )
            return layer(x)
        
        mem = measure_memory_peak(test_fn)
        print(f"{cls.__name__:<20} {mem['baseline_mb']:>12.2f} {mem['peak_mb']:>12.2f} {mem['delta_mb']:>12.2f}")
    
    print("-" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ARNOLD polynomial benchmark suite")
    parser.add_argument("--degree", type=int, default=32, help="Polynomial degree")
    parser.add_argument("--repeats", type=int, default=50, help="Number of timing repeats")
    parser.add_argument("--input-dim", type=int, default=16, help="Input feature dimension")
    parser.add_argument("--batch", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--mode", 
        choices=["basis", "forward", "memory", "hardware-info"], 
        default="basis",
        help="Benchmark mode"
    )
    parser.add_argument(
        "--include-promote", 
        action="store_true", 
        help="Keep promote_to_float64 enabled"
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default=None,
        help="Set mixed-precision policy (e.g., 'mixed_float16', 'mixed_bfloat16')"
    )
    args = parser.parse_args()
    
    # Set mixed precision policy if specified
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy(args.mixed_precision)
        print(f"Mixed-precision policy set to: {args.mixed_precision}")
    
    # Hardware info mode
    if args.mode == "hardware-info":
        print_hardware_info()
        return
    
    # Memory profiling mode
    if args.mode == "memory":
        run_memory_benchmark(args)
        return
    
    # Timing benchmarks (basis or forward)
    x = tf.random.uniform((args.batch, args.input_dim), minval=-1.0, maxval=1.0, dtype=tf.float32)
    families = [
        (Legendre, {"input_clip": (-1.0, 1.0)}),
        (Chebyshev1st, {"input_clip": (-1.0, 1.0)}),
        (Chebyshev2nd, {"input_clip": (-1.0, 1.0)}),
        (Chebyshev3rd, {"input_clip": (-1.0, 1.0)}),
        (Chebyshev4th, {"input_clip": (-1.0, 1.0)}),
        (Jacobi, {"input_clip": (-1.0, 1.0), "alpha_init": 0.5, "beta_init": -0.25}),
        (Gegenbauer, {"input_clip": (-1.0, 1.0), "alpha_init": 0.6}),
        (Hermite, {"input_clip": None}),
        (GeneralizedLaguerre, {"input_clip": (0.0, 4.0), "alpha_init": 0.3}),
        (Bessel, {"input_clip": None}),
        (BannaiIto, {"input_clip": None}),
        (Charlier, {"input_clip": None, "a_init": 1.0}),
    ]
    
    hw = detect_hardware()
    print(f"Hardware: {hw}")
    print(f"Benchmark over {args.repeats} runs, degree={args.degree}, input_dim={args.input_dim}, batch={args.batch}")
    print(f"Mode: {args.mode}")
    print(f"Mixed-precision policy: {tf.keras.mixed_precision.global_policy().name}")
    print("-" * 60)
    
    for cls, extra in families:
        layer = cls(degree=args.degree, units=4, promote_to_float64=args.include_promote, **extra)
        # build
        _ = layer(x)
        if args.mode == "basis":
            pseudo_t = benchmark_basis(layer, x, args.repeats, method="pseudo")
            clenshaw_t = benchmark_basis(layer, x, args.repeats, method="clenshaw")
            print(f"{cls.__name__:<20} pseudo: {pseudo_t*1e3:6.2f} ms | clenshaw: {clenshaw_t*1e3:6.2f} ms")
        else:
            fwd_t = benchmark_forward(layer, x, args.repeats)
            print(f"{cls.__name__:<20} forward: {fwd_t*1e3:6.2f} ms")


if __name__ == "__main__":
    tf.config.experimental_run_functions_eagerly(False)
    main()
