# Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
"""
Tests for the ARNOLD layer registry.
"""

import pytest

from arnold.layers.core.registry import (
    LAYER_REGISTRY,
    LAYER_CATEGORIES,
    get_layer,
    get_layer_class,
    list_layers,
    list_layers_by_category,
    is_registered,
    get_aliases,
)
from arnold.layers.core.kan_base import KANBase


class TestLayerRegistry:
    """Tests for LAYER_REGISTRY dictionary."""

    def test_registry_is_not_empty(self):
        """Registry should contain many layers."""
        assert len(LAYER_REGISTRY) > 50

    def test_all_values_are_kan_subclasses(self):
        """All registered classes should inherit from KANBase."""
        for name, cls in LAYER_REGISTRY.items():
            assert issubclass(cls, KANBase), f"{name} -> {cls} is not a KANBase subclass"

    def test_all_keys_are_lowercase(self):
        """All registry keys should be lowercase."""
        for name in LAYER_REGISTRY.keys():
            assert name == name.lower(), f"Key '{name}' is not lowercase"

    def test_no_hyphens_in_keys(self):
        """Registry keys should use underscores, not hyphens."""
        for name in LAYER_REGISTRY.keys():
            assert "-" not in name, f"Key '{name}' contains hyphen"


class TestGetLayer:
    """Tests for get_layer function."""

    @pytest.mark.parametrize("name,kwargs", [
        ("legendre", {"degree": 3, "units": 8}),
        ("chebyshev", {"degree": 3, "units": 8}),
        ("gegenbauer", {"degree": 3, "units": 8}),
        ("hermite", {"degree": 3, "units": 8}),
        ("laguerre", {"degree": 3, "units": 8}),
        ("fibonacci", {"degree": 3, "units": 8}),
        ("gaussian_rbf", {"units": 8}),
        ("haar", {"units": 8}),
        ("bspline", {"units": 8}),
    ])
    def test_get_layer_basic(self, name, kwargs):
        """Basic layer instantiation works."""
        layer = get_layer(name, **kwargs)
        assert isinstance(layer, KANBase)

    def test_get_layer_case_insensitive(self):
        """Layer names are case-insensitive."""
        layer1 = get_layer("Legendre", degree=3, units=8)
        layer2 = get_layer("LEGENDRE", degree=3, units=8)
        layer3 = get_layer("legendre", degree=3, units=8)
        assert type(layer1) == type(layer2) == type(layer3)

    def test_get_layer_with_hyphen(self):
        """Hyphens are normalized to underscores."""
        layer1 = get_layer("gaussian-rbf", units=8)
        layer2 = get_layer("gaussian_rbf", units=8)
        assert type(layer1) == type(layer2)

    def test_get_layer_with_spaces(self):
        """Spaces are normalized to underscores."""
        layer1 = get_layer("gaussian rbf", units=8)
        layer2 = get_layer("gaussian_rbf", units=8)
        assert type(layer1) == type(layer2)

    def test_get_layer_unknown_raises(self):
        """Unknown layer names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown layer"):
            get_layer("nonexistent_layer", degree=3, units=8)

    def test_get_layer_alias_chebyshev(self):
        """Chebyshev aliases work correctly."""
        layer1 = get_layer("chebyshev", degree=3, units=8)
        layer2 = get_layer("chebyshev_t", degree=3, units=8)
        layer3 = get_layer("chebyshev1", degree=3, units=8)
        assert type(layer1) == type(layer2) == type(layer3)

    def test_get_layer_alias_ultraspherical(self):
        """Ultraspherical is an alias for Gegenbauer."""
        layer1 = get_layer("ultraspherical", degree=3, units=8)
        layer2 = get_layer("gegenbauer", degree=3, units=8)
        assert type(layer1) == type(layer2)

    def test_get_layer_alias_mexican_hat(self):
        """Mexican hat is an alias for Ricker."""
        layer1 = get_layer("mexican_hat", units=8)
        layer2 = get_layer("ricker", units=8)
        assert type(layer1) == type(layer2)


class TestGetLayerClass:
    """Tests for get_layer_class function."""

    def test_get_layer_class_returns_class(self):
        """get_layer_class returns a class, not an instance."""
        cls = get_layer_class("legendre")
        assert isinstance(cls, type)
        assert issubclass(cls, KANBase)

    def test_get_layer_class_instantiation(self):
        """Returned class can be instantiated."""
        cls = get_layer_class("legendre")
        layer = cls(degree=3, units=8)
        assert isinstance(layer, KANBase)

    def test_get_layer_class_unknown_raises(self):
        """Unknown layer names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown layer"):
            get_layer_class("nonexistent_layer")


class TestListLayers:
    """Tests for list_layers function."""

    def test_list_layers_returns_list(self):
        """Returns a list of strings."""
        layers = list_layers()
        assert isinstance(layers, list)
        assert all(isinstance(name, str) for name in layers)

    def test_list_layers_sorted(self):
        """Returned list is sorted."""
        layers = list_layers()
        assert layers == sorted(layers)

    def test_list_layers_canonical_smaller(self):
        """Canonical names (no aliases) are fewer than with aliases."""
        canonical = list_layers(include_aliases=False)
        with_aliases = list_layers(include_aliases=True)
        assert len(canonical) < len(with_aliases)

    def test_list_layers_canonical_has_common_names(self):
        """Common layer names are in canonical list."""
        canonical = list_layers(include_aliases=False)
        # At least these should be present
        assert "legendre" in canonical
        assert "gaussian_rbf" in canonical or "gaussian" in canonical
        assert "haar" in canonical


class TestListLayersByCategory:
    """Tests for list_layers_by_category function."""

    def test_returns_dict(self):
        """Returns a dictionary."""
        cats = list_layers_by_category()
        assert isinstance(cats, dict)

    def test_has_expected_categories(self):
        """Expected categories are present."""
        cats = list_layers_by_category()
        expected = {"polynomial", "rbf", "wavelet", "spline"}
        assert expected.issubset(cats.keys())

    def test_category_values_are_lists(self):
        """Category values are lists of strings."""
        cats = list_layers_by_category()
        for cat, names in cats.items():
            assert isinstance(names, list), f"Category {cat} is not a list"
            assert all(isinstance(n, str) for n in names)

    def test_polynomial_category_has_layers(self):
        """Polynomial category has expected layers."""
        cats = list_layers_by_category()
        poly = cats.get("polynomial", [])
        assert "legendre" in poly
        assert "jacobi" in poly
        assert "chebyshev1" in poly


class TestIsRegistered:
    """Tests for is_registered function."""

    def test_registered_layer(self):
        """Known layers return True."""
        assert is_registered("legendre")
        assert is_registered("chebyshev")
        assert is_registered("gaussian_rbf")

    def test_unregistered_layer(self):
        """Unknown layers return False."""
        assert not is_registered("nonexistent_layer")
        assert not is_registered("xyz_abc")

    def test_case_insensitive(self):
        """Check is case-insensitive."""
        assert is_registered("Legendre")
        assert is_registered("LEGENDRE")


class TestGetAliases:
    """Tests for get_aliases function."""

    def test_gegenbauer_has_ultraspherical_alias(self):
        """Gegenbauer has 'ultraspherical' as alias."""
        from arnold.layers.core.polynomial.orthogonal import Gegenbauer
        aliases = get_aliases(Gegenbauer)
        assert "gegenbauer" in aliases
        assert "ultraspherical" in aliases

    def test_chebyshev1_has_multiple_aliases(self):
        """Chebyshev1st has multiple aliases."""
        from arnold.layers.core.polynomial.orthogonal import Chebyshev1st
        aliases = get_aliases(Chebyshev1st)
        assert len(aliases) >= 3  # chebyshev, chebyshev1, chebyshev_t, etc.

    def test_aliases_are_sorted(self):
        """Aliases are returned sorted."""
        from arnold.layers.core.polynomial.orthogonal import Gegenbauer
        aliases = get_aliases(Gegenbauer)
        assert aliases == sorted(aliases)


class TestLayerCategories:
    """Tests for LAYER_CATEGORIES dictionary."""

    def test_all_category_layers_are_registered(self):
        """All layers mentioned in categories are in registry."""
        for cat, names in LAYER_CATEGORIES.items():
            for name in names:
                assert name in LAYER_REGISTRY, (
                    f"Layer '{name}' in category '{cat}' is not registered"
                )


class TestQOrthogonalLayers:
    """Tests specifically for q-orthogonal polynomial layers."""

    @pytest.mark.parametrize("name", [
        "q_hahn",
        "big_q_jacobi",
        "little_q_jacobi",
        "q_meixner",
        "q_krawtchouk",
        "q_charlier",
        "q_racah",
        "dual_q_hahn",
        "continuous_q_hermite",
        "continuous_q_jacobi",
        "continuous_q_legendre",
    ])
    def test_q_polynomial_instantiation(self, name):
        """Q-polynomial layers can be instantiated."""
        layer = get_layer(name, degree=3, units=8)
        assert isinstance(layer, KANBase)


class TestWaveletLayers:
    """Tests specifically for wavelet layers."""

    @pytest.mark.parametrize("name", [
        "haar",
        "daubechies",
        "symlet",
        "coiflet",
        "ricker",
        "morlet",
        "shannon",
        "meyer",
    ])
    def test_wavelet_instantiation(self, name):
        """Wavelet layers can be instantiated."""
        layer = get_layer(name, units=8)
        assert isinstance(layer, KANBase)


class TestRBFLayers:
    """Tests specifically for RBF layers."""

    @pytest.mark.parametrize("name", [
        "gaussian_rbf",
        "multiquadric",
        "inverse_multiquadric",
        "thin_plate_spline",
        "cauchy",
        "exponential",
    ])
    def test_rbf_instantiation(self, name):
        """RBF layers can be instantiated."""
        layer = get_layer(name, units=8)
        assert isinstance(layer, KANBase)
