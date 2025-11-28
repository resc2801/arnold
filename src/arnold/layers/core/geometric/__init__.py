# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Geometric basis function layers for ARNOLD.

This package provides KAN layers using geometric basis functions,
including spherical harmonics, hyperspherical harmonics, and Zernike polynomials.

Classes
-------
GeometricBase
    Abstract base class for geometric KAN layers.
Zernike
    Zernike polynomials on the unit disk (optics, wavefront analysis).
SphericalHarmonics
    Spherical harmonics Y_l^m(θ, φ) for 3D rotational equivariance.
HypersphericalHarmonics
    Generalized harmonics for n-dimensional spheres (point clouds, molecules).

Mathematical Background
-----------------------
Geometric bases are eigenfunctions of Laplace-Beltrami operators on geometric
domains (disk, sphere, hypersphere). They provide:

- **Rotational equivariance**: Natural for 3D/nD data
- **Orthogonality**: On the geometric domain with appropriate measure
- **Completeness**: Any square-integrable function can be expanded

Applications:
- Optics & wavefront analysis (Zernike)
- 3D point cloud processing (Spherical Harmonics)
- Molecular property prediction (Hyperspherical Harmonics)
- SE(3)-equivariant neural networks

Example
-------
>>> from arnold.layers.core.geometric import Zernike, SphericalHarmonics
>>> import tensorflow as tf
>>>
>>> # Zernike for optical aberrations
>>> zernike_layer = Zernike(degree=8, units=32)
>>>
>>> # Spherical harmonics for 3D data
>>> sh_layer = SphericalHarmonics(max_degree=4, units=64)
"""

from arnold.layers.core.geometric.base import GeometricBase
from arnold.layers.core.geometric.zernike import Zernike
from arnold.layers.core.geometric.spherical_harmonics import SphericalHarmonics
from arnold.layers.core.geometric.hyperspherical_harmonics import HypersphericalHarmonics

__all__ = [
    # Base class
    "GeometricBase",
    # Disk-based
    "Zernike",
    # Sphere-based
    "SphericalHarmonics",
    "HypersphericalHarmonics",
]
