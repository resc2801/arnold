# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
r"""
Spline-based KAN layers.

This package provides spline layers for Kolmogorov-Arnold Networks:
- BSpline: Cox-de Boor B-splines with configurable order
- CatmullRom: Cubic interpolating spline (C¹ continuity)
- Cardinal: Generalized spline with adjustable tension
"""
from arnold.layers.core.splines.base import SplineBase
from arnold.layers.core.splines.bspline import BSpline
from arnold.layers.core.splines.cardinal import Cardinal
from arnold.layers.core.splines.catmull_rom import CatmullRom


__all__ = ["SplineBase", "BSpline", "Cardinal", "CatmullRom"]
