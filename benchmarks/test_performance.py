# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""
Performance Benchmark Suite for ARNOLD KAN Layers

This module provides reproducible benchmarks to track performance regressions.
Results can be compared across commits to catch performance degradation.

Usage:
    python -m pytest benchmarks/test_performance.py -v --benchmark-autosave
    python -m pytest benchmarks/test_performance.py --benchmark-compare

Requires pytest-benchmark: pip install pytest-benchmark
"""

import numpy as np
import pytest
import tensorflow as tf

from arnold.layers import (
    Chebyshev1st,
    GaussianRBF,
    Legendre,
    Ricker,
)


# Skip if pytest-benchmark not installed
pytest_benchmark = pytest.importorskip("pytest_benchmark")


class TestPolynomialBenchmarks:
    """Benchmark polynomial layer forward pass performance."""

    @pytest.fixture
    def sample_input_small(self):
        """Small batch for latency testing."""
        return tf.constant(np.random.randn(32, 8).astype(np.float32))

    @pytest.fixture
    def sample_input_medium(self):
        """Medium batch for throughput testing."""
        return tf.constant(np.random.randn(256, 16).astype(np.float32))

    @pytest.fixture
    def sample_input_large(self):
        """Large batch for peak throughput."""
        return tf.constant(np.random.randn(1024, 32).astype(np.float32))

    def test_legendre_degree5_small_batch(self, benchmark, sample_input_small):
        """Benchmark Legendre degree=5 with small batch."""
        layer = Legendre(degree=5, units=16)
        layer(sample_input_small)  # Warm up / build
        
        @tf.function(jit_compile=True)
        def forward(x):
            return layer(x)
        
        # Warm up JIT
        forward(sample_input_small)
        
        result = benchmark(lambda: forward(sample_input_small))

    def test_legendre_degree5_medium_batch(self, benchmark, sample_input_medium):
        """Benchmark Legendre degree=5 with medium batch."""
        layer = Legendre(degree=5, units=32)
        layer(sample_input_medium)
        
        @tf.function(jit_compile=True)
        def forward(x):
            return layer(x)
        
        forward(sample_input_medium)
        result = benchmark(lambda: forward(sample_input_medium))

    def test_legendre_degree20_medium_batch(self, benchmark, sample_input_medium):
        """Benchmark Legendre degree=20 (high degree) with medium batch."""
        layer = Legendre(degree=20, units=32)
        layer(sample_input_medium)
        
        @tf.function(jit_compile=True)
        def forward(x):
            return layer(x)
        
        forward(sample_input_medium)
        result = benchmark(lambda: forward(sample_input_medium))

    def test_chebyshev_degree10_large_batch(self, benchmark, sample_input_large):
        """Benchmark Chebyshev degree=10 with large batch (throughput test)."""
        layer = Chebyshev1st(degree=10, units=64)
        layer(sample_input_large)
        
        @tf.function(jit_compile=True)
        def forward(x):
            return layer(x)
        
        forward(sample_input_large)
        result = benchmark(lambda: forward(sample_input_large))


class TestRBFBenchmarks:
    """Benchmark RBF layer forward pass performance."""

    @pytest.fixture
    def sample_input_medium(self):
        return tf.constant(np.random.randn(256, 16).astype(np.float32))

    def test_gaussian_rbf_grids10(self, benchmark, sample_input_medium):
        """Benchmark GaussianRBF with 10 grid points."""
        layer = GaussianRBF(num_grids=10, units=32)
        layer(sample_input_medium)
        
        @tf.function(jit_compile=True)
        def forward(x):
            return layer(x)
        
        forward(sample_input_medium)
        result = benchmark(lambda: forward(sample_input_medium))

    def test_gaussian_rbf_grids50(self, benchmark, sample_input_medium):
        """Benchmark GaussianRBF with 50 grid points (high resolution)."""
        layer = GaussianRBF(num_grids=50, units=32)
        layer(sample_input_medium)
        
        @tf.function(jit_compile=True)
        def forward(x):
            return layer(x)
        
        forward(sample_input_medium)
        result = benchmark(lambda: forward(sample_input_medium))


class TestWaveletBenchmarks:
    """Benchmark wavelet layer forward pass performance."""

    @pytest.fixture
    def sample_input_medium(self):
        return tf.constant(np.random.randn(256, 16).astype(np.float32))

    def test_ricker_grids10(self, benchmark, sample_input_medium):
        """Benchmark Ricker wavelet with 10 grid points."""
        layer = Ricker(num_grids=10, units=32)
        layer(sample_input_medium)
        
        @tf.function(jit_compile=True)
        def forward(x):
            return layer(x)
        
        forward(sample_input_medium)
        result = benchmark(lambda: forward(sample_input_medium))


class TestTrainingBenchmarks:
    """Benchmark training step performance (forward + backward)."""

    @pytest.fixture
    def training_data(self):
        x = tf.constant(np.random.randn(128, 16).astype(np.float32))
        y = tf.constant(np.random.randn(128, 1).astype(np.float32))
        return x, y

    def test_legendre_training_step(self, benchmark, training_data):
        """Benchmark single training step with Legendre layer."""
        x, y = training_data
        
        model = tf.keras.Sequential([
            Legendre(degree=5, units=32),
            Legendre(degree=5, units=1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model(x)  # Build
        
        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                pred = model(x, training=True)
                loss = tf.reduce_mean((pred - y) ** 2)
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss
        
        # Warm up
        train_step(x, y)
        
        result = benchmark(lambda: train_step(x, y))

    def test_mixed_layers_training_step(self, benchmark, training_data):
        """Benchmark training step with mixed layer types."""
        x, y = training_data
        
        model = tf.keras.Sequential([
            Legendre(degree=5, units=32),
            GaussianRBF(num_grids=10, units=16),
            Chebyshev1st(degree=3, units=1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model(x)
        
        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                pred = model(x, training=True)
                loss = tf.reduce_mean((pred - y) ** 2)
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss
        
        train_step(x, y)
        result = benchmark(lambda: train_step(x, y))


class TestMemoryBenchmarks:
    """Track memory usage patterns (informational, not strict benchmarks)."""

    def test_high_degree_memory(self):
        """Ensure high-degree layers don't OOM on reasonable batch sizes."""
        layer = Legendre(degree=50, units=64)
        
        # Should complete without OOM
        x = tf.constant(np.random.randn(64, 32).astype(np.float32))
        y = layer(x)
        
        # Basic sanity check
        assert y.shape == (64, 64)
        assert not tf.reduce_any(tf.math.is_nan(y))

    def test_tucker_decomposition_memory(self):
        """Tucker decomposition should reduce memory footprint."""
        # Full tensor: 16 * 11 * 64 = 11,264 params
        # Tucker (4,4,4): 4*4*4 + 16*4 + 11*4 + 64*4 = 64 + 64 + 44 + 256 = 428 params
        layer_full = Legendre(degree=10, units=64)
        layer_tucker = Legendre(degree=10, units=64, core_ranks=(4, 4, 4))
        
        x = tf.constant(np.random.randn(8, 16).astype(np.float32))
        layer_full(x)
        layer_tucker(x)
        
        full_params = sum(np.prod(v.shape) for v in layer_full.trainable_variables)
        tucker_params = sum(np.prod(v.shape) for v in layer_tucker.trainable_variables)
        
        # Tucker should have fewer parameters
        assert tucker_params < full_params, f"Tucker {tucker_params} >= Full {full_params}"


# Baseline performance targets (update after establishing baseline)
PERFORMANCE_TARGETS = {
    "legendre_degree5_small": {"max_time_ms": 10},
    "legendre_degree5_medium": {"max_time_ms": 20},
    "chebyshev_degree10_large": {"max_time_ms": 50},
}


if __name__ == "__main__":
    # Run benchmarks with comparison
    pytest.main([
        __file__, 
        "-v",
        "--benchmark-autosave",
        "--benchmark-group-by=func",
    ])
