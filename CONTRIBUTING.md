# Contributing to ARNOLD

Thank you for your interest in contributing to ARNOLD! This guide will help you get started.

## Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/resc2801/arnold.git
cd arnold
```

### 2. Create a virtual environment

We recommend using conda:

```bash
conda create -n arnold-dev python=3.11
conda activate arnold-dev
```

### 3. Install in development mode

```bash
pip install -e ".[dev]"
```

Or using make:

```bash
make install-dev
```

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run specific test file
pytest tests/test_polynomial_correctness.py -v

# Run specific test class
pytest tests/test_polynomial_correctness.py::TestLegendre -v
```

### Code Quality

```bash
# Format code with black
make format

# Run linter
make lint

# Type checking (if configured)
mypy src/arnold
```

### Building Documentation

```bash
cd docs
make html
# Open docs/_build/html/index.html in your browser
```

## Code Style

### Python Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Maximum line length: 120 characters

### Docstring Format

We use **NumPy-style docstrings** for all public APIs:

```python
def example_function(param1: int, param2: float = 1.0) -> tf.Tensor:
    """
    Short description of the function.

    Longer description if needed, explaining the function's purpose
    and any important details.

    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : float, default 1.0
        Description of param2.

    Returns
    -------
    tf.Tensor
        Description of the return value.

    Raises
    ------
    ValueError
        When param1 is negative.

    Examples
    --------
    >>> result = example_function(5, 2.0)
    >>> print(result.shape)
    (5,)

    Notes
    -----
    Additional implementation notes or mathematical details.

    References
    ----------
    .. [1] Author, "Title", Journal, Year.
    """
```

### Layer Implementation Guidelines

When implementing a new KAN layer:

1. **Inherit from the appropriate base class:**
   - `PolynomialBase` for polynomial-based layers
   - `RBFBase` for radial basis function layers
   - `WaveletBase` for wavelet-based layers

2. **Implement required methods:**
   - `pseudo_vandermonde(x)` for polynomials
   - `get_kernels(r)` for RBFs
   - `mother_wavelet(x)` for wavelets

3. **Add Keras serialization:**
   ```python
   @tfk.utils.register_keras_serializable(package="arnold", name="YourLayer")
   class YourLayer(PolynomialBase):
       ...
   ```

4. **Handle numerical stability:**
   - Use `input_clip` for domain restrictions
   - Consider `promote_to_float64` for high-degree polynomials
   - Use `arnold.utils.constants.PARAM_EPS` for small epsilon values

5. **Add tests:**
   - Shape tests in `test_layer_properties.py`
   - Correctness tests comparing to reference implementations
   - Gradient tests for trainable parameters

## Adding a New Polynomial Layer

### Example: Adding a new orthogonal polynomial

```python
# In src/arnold/layers/core/polynomial/orthogonal.py

@tfk.utils.register_keras_serializable(package="arnold", name="NewPolynomial")
class NewPolynomial(PolynomialBase):
    r"""
    Kolmogorov-Arnold Network layer using NewPolynomial polynomials.

    The NewPolynomial polynomials satisfy the three-term recurrence:

    .. math::
        P_{n+1}(x) = \alpha_n x P_n(x) + \beta_n P_{n-1}(x)

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.
    units : int
        Output dimensionality.
    alpha_init : float | None
        Initial value for alpha parameter.

    Notes
    -----
    Stable on :math:`[-1, 1]}; use ``input_clip=(-1, 1)`` for inputs
    outside this range.
    """

    def __init__(
        self,
        degree: int,
        *,
        units: int,
        alpha_init: float | None = None,
        alpha_trainable: bool = True,
        input_clip: tuple[float, float] | None = (-1.0, 1.0),
        **kwargs,
    ):
        super().__init__(degree=degree, units=units, input_clip=input_clip, **kwargs)
        self.alpha_init = alpha_init
        self.alpha_trainable = alpha_trainable
        self.alpha = None

    def build(self, input_shape):
        super().build(input_shape)
        self.alpha = create_trainable_param(
            self, "alpha", self.alpha_init, trainable=self.alpha_trainable
        )

    @kan_fn
    def pseudo_vandermonde(self, x):
        # Implement the recurrence relation
        basis = [tf.ones_like(x)]
        if self.degree > 0:
            basis.append(self.alpha * x)
        for n in range(2, self.degree + 1):
            basis.append(
                2 * self.alpha * x * basis[n-1] - basis[n-2]
            )
        return tf.stack(basis, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha_init": self.alpha_init,
            "alpha_trainable": self.alpha_trainable,
        })
        return config
```

## Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Write tests** for your changes
3. **Update documentation** if needed
4. **Run the test suite** to ensure all tests pass
5. **Submit a pull request** with a clear description

### PR Checklist

- [ ] Tests pass locally (`make test`)
- [ ] Code is formatted (`make format`)
- [ ] Docstrings follow NumPy style
- [ ] New features are documented
- [ ] CHANGELOG.md is updated (for significant changes)

## Reporting Issues

When reporting issues, please include:

1. **Python version** (`python --version`)
2. **TensorFlow version** (`python -c "import tensorflow; print(tensorflow.__version__)"`)
3. **ARNOLD version** (`python -c "import arnold; print(arnold.__version__)"`)
4. **Minimal reproducible example**
5. **Full error traceback**

## Questions?

- Open a [GitHub Issue](https://github.com/resc2801/arnold/issues) for bugs
- Start a [GitHub Discussion](https://github.com/resc2801/arnold/discussions) for questions

Thank you for contributing! 🎉
