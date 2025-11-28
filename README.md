# $\mathtt{ARNOLD}$ 

[![PyPI version](https://badge.fury.io/py/arnold-kan.svg)](https://badge.fury.io/py/arnold-kan)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.16+](https://img.shields.io/badge/tensorflow-2.16+-orange.svg)](https://www.tensorflow.org/)
[![License: Academic/Non-Commercial](https://img.shields.io/badge/License-Academic%2FNon--Commercial-red.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://arnold-kan.readthedocs.io/)
[![Tests](https://img.shields.io/badge/tests-1428%20passing-brightgreen.svg)](#)

$\mathtt{ARNOLD}$ offers `tf.keras` implementations of [Kolmogorov-Arnold Networks (KAN) layers](https://arxiv.org/pdf/2404.19756?trk=public_post_main-feed-card-text) using various basis functions.

**80+ layer implementations** across 9 basis families: polynomial, q-orthogonal, spectral, geometric, special, RBF, wavelet, spline, and more.

Kolmogorov-Arnold Networks (KANs) are a type of neural network inspired by a mathematical theorem related to the representation of continuous functions. 

The Kolmogorov-Arnold representation theorem states that any multivariate continuous function can be represented as a superposition of continuous functions of a single variable and addition. Leveraging this theorem, KANs are designed to approximate complex multivariate functions by breaking them down into simpler, univariate functions.



## Installation
Install $\mathtt{ARNOLD}$ via pip

```shell
pip install arnold-kan
```

## Quick Start 
Simply use $\mathtt{ARNOLD}$'s KAN layers as a drop-in-replacement for `tf.keras.layers.Dense` (use `units` for output size) and mix with any standard layers.

```python
import tensorflow as tf
from arnold.layers import Chebyshev1st, Legendre, Bump, FourierKAN

tfk = tf.keras
tfkl = tfk.layers

fancy_kan = tfk.Sequential(
    [
        tfkl.Reshape(target_shape=(2, )),
        tfkl.Rescaling(scale=1./127.5, offset=-1),
        Chebyshev1st(degree=2, units=8, input_clip=(-1, 1)),
        tfkl.LayerNormalization(),
        Legendre(degree=3, units=6, input_clip=(-1, 1)),
        tfkl.LayerNormalization(),
        FourierKAN(degree=4, units=4),  # NEW: Spectral basis
        tfkl.LayerNormalization(),
        Bump(units=1, input_clip=(-2, 2)),
        tfkl.Activation(tfk.activations.sigmoid)
    ],
    name="fancy_kan"
)

# Minimal end-to-end example
x = tf.random.uniform((32, 2), minval=-1.0, maxval=1.0)
model = tfk.Sequential([Chebyshev1st(degree=3, units=4, input_clip=(-1, 1)), tfkl.Dense(1)])
model.compile(optimizer="adam", loss="mse")
model.fit(x, tf.random.uniform((32, 1)), epochs=2, verbose=0)

# Inference
preds = model(tf.random.uniform((4, 2), minval=-1.0, maxval=1.0))
print(preds.shape)  # (4, 1)
```

### Layer Registry (NEW!)

Use string-based layer creation with the centralized registry:

```python
from arnold.layers.core.registry import get_layer, list_layers, list_layers_by_category

# Create layers by name
layer = get_layer("legendre", degree=5, units=32)
layer = get_layer("fourier", degree=4, units=16)
layer = get_layer("gaussian_rbf", units=8, grid_size=10)

# List all 130+ available layers
print(list_layers())  # ['airy', 'askey_wilson', 'bessel', ...]

# Browse by category
categories = list_layers_by_category()
print(categories.keys())  # ['polynomial', 'q_orthogonal', 'spectral', 'geometric', ...]
```

Notes:
- `units` replaces legacy `output_dim`; weights are created in `build()` so shapes are inferred.
- Use `input_clip` to keep inputs in the canonical domain (e.g., `(-1, 1)` for most orthogonal polynomials).

### Basis domain tips
- Orthogonal polynomials (Legendre, Chebyshev, Gegenbauer, Jacobi): keep inputs in ``[-1, 1]`` (use `input_clip=(-1, 1)` or preprocessing).
- Laguerre: defined on ``[0, ∞)``; ensure inputs are non-negative or clip.
- Hermite/Bessel/Laurent: on ``ℝ``; consider clipping or scaling for large magnitudes; Laurent avoids poles via internal clamp.
- RBFs: control radii with ``grid_min/grid_max``; shape params are positive via softplus.
- Wavelets: scales are positive via softplus; use ``input_clip`` to bound inputs if needed.
- Spectral (Fourier, RFF): work on ``ℝ`` but periodic signals benefit from proper normalization.
- Geometric (Zernike, Spherical): inputs should be in appropriate domains (unit disk, sphere).
- Performance: high-degree polynomial bases use scan/Clenshaw recurrences to reduce ops and memory vs. materializing the full basis; set a sensible ``degree`` to balance capacity and speed.
- Benchmarking: run ``make bench`` to compare pseudo vs. scan/Clenshaw for common bases.

## Examples

| Task | Dataset |  |
| :- | :- | :-: |
| Multinomial Classifcation             | MNIST                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/resc2801/arnold/blob/main/examples/mnist/mnist.ipynb)   |
| Binary classification                 | "Two moons"             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/resc2801/arnold/blob/main/examples/two_moons/two_moons.ipynb)   |
| Multivariate Function Interpolation   | 2D function             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/resc2801/arnold/blob/main/examples/multivariate_interpolation/fractal.ipynb)   |
| Multivariate Function Interpolation   | 2D Helmholtz equation   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/resc2801/arnold/blob/main/examples/multivariate_interpolation/helmholtz.ipynb)   |

## Available KAN Layers

$\mathtt{ARNOLD}$ provides **80+ layer implementations** organized into 9 basis families:

| Category | Layers | Description |
|:---------|:-------|:------------|
| Polynomial | 20+ | Orthogonal polynomials (Legendre, Chebyshev, Jacobi, etc.) |
| q-Orthogonal | 19 | Quantum group polynomials (q-Hahn, q-Racah, q-Hermite, etc.) |
| Sequences | 14 | Fibonacci-type and Lucas-type polynomial sequences |
| Spectral | 2+ | Fourier, Random Fourier Features |
| Geometric | 3 | Zernike, Spherical Harmonics, Hyperspherical Harmonics |
| Special | 2+ | Airy, Bessel functions (+ 6 stubs) |
| RBF | 11 | Gaussian, Multiquadric, Thin-plate spline, etc. |
| Wavelet | 12 | Haar, Daubechies, Morlet, Mexican Hat, etc. |
| Spline | 4 | B-spline, Catmull-Rom, Cardinal, Hermite |

### Polynomial bases

In the context of KANs, polynomial bases can be employed to represent the univariate functions that constitute the network. When incorporating polynomial bases into KANs, the network's layers are designed to transform the input variables into polynomial functions. These polynomial functions serve as the univariate components specified by the Kolmogorov-Arnold representation theorem. By using polynomials, the network can efficiently approximate smooth, continuous functions, exploiting the well-known properties of polynomials such as their ability to be easily differentiated and integrated.

#### Continuous orthogonal polynomials

Orthogonal polynomials, such as Legendre, Chebyshev, and Hermite polynomials, have coefficients that are uncorrelated when integrated over a certain range with a specific weight function.
The orthogonality condition ensures that each polynomial basis function captures unique aspects of the data, minimizing overlap and improving the network's learning efficiency.

| Layer                                                                                             | Definition | Parameters | Support | Implementation |
| :-                                                                                                | :- | :- | :- | :- | 
| [Al-Salam-Carlitz (1st kind)](src/arnold/layers/core/polynomial/orthogonal/al_salam_carlitz.py)        | $U^{(a)}_{n+1} (x;q) = (x - (1 + a) q^{n}) U^{(a)}_{n} (x;q) + a q^{n-1} (1 - q^{n}) U^{(a)}_{n-1} (x;q)$ | $a, q$ | $\mathbb{R}$ | three-term recurence |
| [Al-Salam-Carlitz (2nd kind)](src/arnold/layers/core/polynomial/orthogonal/al_salam_carlitz.py)        | $V^{a}_{n+1} (x; q) = U^{a}_{n+1} (x; \frac{1}{q})$ | $a, q$ | $\mathbb{R}$ | three-term recurence |
| [Askey-Wilson](src/arnold/layers/core/polynomial/orthogonal/askey_wilson.py)                           | $p_{n}(x;a,b,c,d\mid q) = a^{-n}(ab,ac,ad;q)_{n} \; {}_{4}{\phi}_{3} \left[\begin{matrix}q^{-n}&abcdq^{n-1}&ae^{i\theta }&ae^{-i\theta }\\ab&ac&ad\end{matrix};q,q\right]$ | $a, b, c, d$ | $\mathbb{R}$ | three-term recurence |
| [Bannai-Ito](src/arnold/layers/core/polynomial/orthogonal/bannai_ito.py)                               | $y_{n}(x)=\sum_{k=0}^{n}{\frac {(n+k)!}{(n-k)!k!}}\,\left({\frac {x}{2}}\right)^{k}$ |- | $\mathbb{R}$ | 3-term recurrence |
| [Bessel](src/arnold/layers/core/polynomial/orthogonal/bessel_family.py)                                       | $y_{n}(x)=\sum_{k=0}^{n}{\frac {(n+k)!}{(n-k)!k!}}\,\left({\frac {x}{2}}\right)^{k}$ |- | $\mathbb{R}$ | 3-term recurrence |
| [Chebyshev (1st kind)](src/arnold/layers/core/polynomial/orthogonal/jacobi_family.py)                      | $T_{n}(\cos \theta )=\cos(n\theta)$ | - | $\mathbb{R}$ | trigonometric |
| [Chebyshev (2nd kind)](src/arnold/layers/core/polynomial/orthogonal/jacobi_family.py)                      | $U_{n}(\cos \theta )\sin \theta =\sin ((n+1)\theta), \; \cos(\theta) = \tfrac{2x - (a+b)}{b-a}, \; \theta \in [0, \pi]$ | $a, b$ | $[a, b]$ | trigonometric |
| [Chebyshev (3rd kind)](src/arnold/layers/core/polynomial/orthogonal/jacobi_family.py)                      | $V_{n}(x) = \tfrac{\cos(n+1/2) \theta}{\cos(\theta/2)}, \; \cos(\theta) = \tfrac{2x - (a+b)}{b-a}, \; \theta \in [0, \pi]$ | $a, b$ | $[a, b]$ | trigonometric |
| [Chebyshev (4th kind)](src/arnold/layers/core/polynomial/orthogonal/jacobi_family.py)                      | $W_{n}(x) = \tfrac{\sin(n+1/2)\theta}{\sin(\theta / 2)}, \; \cos(\theta) = \tfrac{2x - (a+b)}{b-a}, \; \theta \in [0, \pi]$ | $a, b$ | $[a, b]$ | trigonometric |
| [Gegenbauer](src/arnold/layers/core/polynomial/orthogonal/jacobi_family.py)                               | $C_{n}^{\alpha}(x)=\frac {(2\alpha +n-1)!}{(2\alpha -1)! \,n!}\, {}_{2}F_{1}\left(-n,2\alpha +n;\alpha +{\frac {1}{2}}; {\frac {1-x}{2}}\right)$ | $\alpha > -\tfrac{1}{2}$  | $\mathbb{R}$ | 3-term recurence |
| [Physicist's Hermite](src/arnold/layers/core/polynomial/orthogonal/hermite_family.py)                         | $H_{n}(x)=(-1)^{n}e^{x^{2}}{\frac {d^{n}}{dx^{n}}}e^{-x^{2}}$ | - | $\mathbb{R}$ | 3-term recurence |
| [Jacobi](src/arnold/layers/core/polynomial/orthogonal/jacobi_family.py)                                       | $P_{n}^{(\alpha ,\beta )}(x)={\frac {(\alpha +1)_{n}}{n!}} \, {}_{2}F_{1}\left(-n,1+\alpha +\beta +n;\alpha +1;{\tfrac {1}{2}}(1-x)\right)$ | $\alpha, \beta$ | $\mathbb{R}$ | 3-term recurence |
| [Generalized Laguerre](src/arnold/layers/core/polynomial/orthogonal/laguerre_family.py)                       | $L_{n}^{(\alpha )}(x)=\sum _{i=0}^{n}(-1)^{i}{n+\alpha  \choose n-i}{\frac {x^{i}}{i!}}$ | $\alpha$ | $\mathbb{R}$ | 3-term recurence |
| [Legendre](src/arnold/layers/core/polynomial/orthogonal/jacobi_family.py)                                   | $P_{n}(x)=\sum _{k=0}^{\lfloor n/2\rfloor }(-1)^{k}{\frac {(2n-2k)!\ }{(n-k)!\ (n-2k)!\ k!\ 2^{n}}}x^{n-2k}$ | - | $\mathbb{R}$ | 3-term recurence |
| [Associated Meixner-Pollaczek](src/arnold/layers/core/polynomial/orthogonal/meixner_pollaczek.py)      | $(n + c + 1) P^{\lambda}_{n+1}(x; \phi, c) = 2x \sin(\phi) + 2(n + c + \lambda) P^{\lambda}_{n}(x; \phi, c) - (n + c + 2\lambda - 1) P^{\lambda}_{n-1}(x; \phi, c)$ | $c, \lambda, \phi$ | $\mathbb{R}$ | 3-term recurence |
| [Pollaczek](src/arnold/layers/core/polynomial/orthogonal/pollaczek.py)                                 | $n P_{n}(x;a,b) = ((2n-1+2a)x+2b)P_{n-1}(x;a,b)-(n-1)P_{n-2}(x;a,b)$ | $a, b$ | $\mathbb{R}$ | 3-term recurence |
| Wilson | - | - | - | TBD |


#### Discrete orthogonal polynomials

Discrete orthogonal polynomial bases are sequences of polynomials that are pairwise orthogonal with respect to a discrete measure.

| Layer                                                                                             | Definition | Parameters | Support | Implementation |
| :-                                                                                                | :- | :- | :- | :- | 
| [Charlier](src/arnold/layers/core/polynomial/orthogonal/charlier.py)                              | $C_n(x; a)$ | $a$ | $\mathbb{N}_0$ | 3-term recurrence |
| Discrete Chebyshev    | - | - | - | - |
| Dual Hahn             | - | - | - | - |
| Hahn                  | - | - | - | - |
| Krawtchouk            | - | - | - | - |
| Meixner               | - | - | - | - |
| Racah                 | - | - | - | - |

#### q-Orthogonal polynomials (19 layers)

Quantum group analogues of classical orthogonal polynomials, essential for quantum mechanics and combinatorics.

| Layer | Parameters | Implementation |
| :- | :- | :- |
| [QHahn](src/arnold/layers/core/polynomial/q_orthogonal/q_hahn.py) | $\alpha, \beta, N, q$ | 3-term recurrence |
| [BigQJacobi](src/arnold/layers/core/polynomial/q_orthogonal/big_q_jacobi.py) | $a, b, c, q$ | 3-term recurrence |
| [LittleQJacobi](src/arnold/layers/core/polynomial/q_orthogonal/little_q_jacobi.py) | $a, b, q$ | 3-term recurrence |
| [QMeixner](src/arnold/layers/core/polynomial/q_orthogonal/q_meixner.py) | $b, c, q$ | 3-term recurrence |
| [QKrawtchouk](src/arnold/layers/core/polynomial/q_orthogonal/q_krawtchouk.py) | $p, N, q$ | 3-term recurrence |
| [QCharlier](src/arnold/layers/core/polynomial/q_orthogonal/q_charlier.py) | $a, q$ | 3-term recurrence |
| [QRacah](src/arnold/layers/core/polynomial/q_orthogonal/q_racah.py) | $\alpha, \beta, \gamma, \delta, q$ | 3-term recurrence |
| [DualQHahn](src/arnold/layers/core/polynomial/q_orthogonal/dual_q_hahn.py) | $\gamma, \delta, N, q$ | 3-term recurrence |
| [DualQKrawtchouk](src/arnold/layers/core/polynomial/q_orthogonal/dual_q_krawtchouk.py) | $c, N, q$ | 3-term recurrence |
| [AffineQKrawtchouk](src/arnold/layers/core/polynomial/q_orthogonal/affine_q_krawtchouk.py) | $p, N, q$ | 3-term recurrence |
| [DiscreteQHermite1](src/arnold/layers/core/polynomial/q_orthogonal/discrete_q_hermite1.py) | $q$ | 3-term recurrence |
| [DiscreteQHermite2](src/arnold/layers/core/polynomial/q_orthogonal/discrete_q_hermite2.py) | $q$ | 3-term recurrence |
| [ContinuousQHermite](src/arnold/layers/core/polynomial/q_orthogonal/continuous_q_hermite.py) | $q$ | 3-term recurrence |
| [ContinuousQJacobi](src/arnold/layers/core/polynomial/q_orthogonal/continuous_q_jacobi.py) | $\alpha, \beta, q$ | 3-term recurrence |
| [ContinuousQUltraspherical](src/arnold/layers/core/polynomial/q_orthogonal/continuous_q_ultraspherical.py) | $\beta, q$ | 3-term recurrence |
| [QuantumQKrawtchouk](src/arnold/layers/core/polynomial/q_orthogonal/quantum_q_krawtchouk.py) | $p, N, q$ | 3-term recurrence |
| [ContinuousQLaguerre](src/arnold/layers/core/polynomial/q_orthogonal/continuous_q_laguerre.py) | $\alpha, q$ | 3-term recurrence |
| [ContinuousQLegendre](src/arnold/layers/core/polynomial/q_orthogonal/continuous_q_legendre.py) | $q$ | 3-term recurrence |

#### Non-orthogonal polynomials

Non-orthogonal polynomials do not satisfy the condition of orthogonality, meaning their inner product is not necessarily zero for distinct polynomials. 
This may allow for a broader selection of basis functions, which can be tailored to fit particular types of data or specific functional forms that might be more challenging to capture with orthogonal polynomials.

| Layer | Definition | Parameters | Support | Implementation |
| :- | :- | :- | :- | :- |
| [Boubaker](src/arnold/layers/core/polynomial/non_orthogonal.py)          | $B_{n}(x)=\sum_{p=0}^{\lfloor n/2 \rfloor} \frac{n-4p}{n-p} \binom{n-p}{p} (-1)^{p} x^{n-2p}$     | - | $\mathbb{R}$ | 3-term recurrence |                     
| [Lucas](src/arnold/layers/core/polynomial/sequences/lucas.py)  | $L_{n}(x)=2^{-n}[(x - \sqrt{x^2+4})^n + (x + \sqrt{x^2+4})^n]$                                    | - | $\mathbb{R}$ | 3-term recurrence |
| [Laurent](src/arnold/layers/core/polynomial/laurent.py)            | ${p(X)=\sum_{k \in \mathbb{Z}} a_{k} X^{k},\quad a_{k} \in \mathbb{R}}$                                    | - | $\mathbb{R}$ | 3-term recurrence |
| Bernstein                                                     | $B_{n}(x) = \sum_{\nu =0}^{n} \beta_{\nu} {\binom{n}{\nu }}x^{\nu}\left(1-x\right)^{n-\nu}$       | - | $[0, 1]$     | TBD |

 
#### Lucas polynomial sequences

A Lucas polynomial sequence is a pair of generalized polynomials which generalize the Lucas sequence to polynomials. We offer a number of special cases.

| Layer                                                                                 | Definition | Parameters | Support | Implementation |
| :-                                                                                    | :- | :- | :- | :- |
| [Chebyshev (1st kind)](src/arnold/layers/core/polynomial/orthogonal/jacobi_family.py)          | $T_{n}(\cos \theta )=\cos(n\theta)$ | - | $\mathbb{R}$ | trigonometric |
| [Chebyshev (2nd kind)](src/arnold/layers/core/polynomial/orthogonal/jacobi_family.py)          | $U_{n}(\cos \theta )\sin \theta =\sin ((n+1)\theta), \; \cos(\theta) = \tfrac{2x - (a+b)}{b-a}, \; \theta \in [0, \pi]$ | $a, b$ | $[a, b]$ | trigonometric |
| [Fermat](src/arnold/layers/core/polynomial/sequences/fermat.py)                        | $F_{n+1}(x) = 3x F_{n}(x) - 2 F_{n-1}(x)$   | - | $\mathbb{R}$ | 3-term recurrence |
| [Fermat-Lucas](src/arnold/layers/core/polynomial/sequences/fermat.py)            | $f_{n+1}(x) = 3x f_{n}(x) - 2 f_{n-1}(x)$   | - | $\mathbb{R}$ | 3-term recurrence |
| [Fibonacci](src/arnold/layers/core/polynomial/sequences/fibonacci.py)                      | $F_{n+2}(x) = x F_{n+1}(x) + F_{n}(x)$ | - | $\mathbb{R}$ | 3-term recurrence |
| [Jacobsthal](src/arnold/layers/core/polynomial/sequences/jacobsthal.py)                | $J_{n+1}(x) = 1 J_{n}(x) + 2x J_{n-1}(x)$   | - | $\mathbb{R}$ | 3-term recurrence |
| [Jacobsthal-Lucas](src/arnold/layers/core/polynomial/sequences/jacobsthal.py)    | $j_{n+1}(x) = 1 j_{n}(x) + 2x j_{n-1}(x)$   | - | $\mathbb{R}$ | 3-term recurrence |
| [Lucas](src/arnold/layers/core/polynomial/sequences/lucas.py)                          | $L_{n+1}(x) = x L_{n}(x) + L_{n-1}(x)$           | - | $\mathbb{R}$ | 3-term recurrence |
| [Pell](src/arnold/layers/core/polynomial/sequences/pell.py)                            | $P_{n+1}(x) = 2 x * P_{n}(x) + P_{n-1}(x)$       | - | $\mathbb{R}$ | 3-term recurrence |
| [Pell-Lucas](src/arnold/layers/core/polynomial/sequences/pell.py)                | $Q_{n+1}(x) = x Q_{n}(x) + Q_{n-1}(x)$           | - | $\mathbb{R}$ | 3-term recurrence |


#### (Generalized) Fibonacci polynomials

The Fibonacci polynomials are a polynomial sequence which can be considered as a generalization of the Fibonacci numbers.
We also provide polynomial sequences based on Fibonacci numbers of higher order. 

| Layer                                                                 | Definition | Parameters | Support | Implementation |
| :-                                                                    | :- | :-: | :-: | :- |
| [Fibonacci](src/arnold/layers/core/polynomial/sequences/fibonacci.py)      | $F_{n+2}(x) = x F_{n+1}(x) + F_{n}(x)$ | - | $\mathbb{R}$ | 3-term recurrence |
| [Tribonacci](src/arnold/layers/core/polynomial/sequences/tribonacci.py)    | $F_{n+3}(x) = x F_{n+2}(x) + \sum_{n}^{n+1} F_{n}(x)$ | - | $\mathbb{R}$ | 4-term recurrence |
| [Tetranacci](src/arnold/layers/core/polynomial/sequences/tetranacci.py)    | $F_{n+4}(x) = x F_{n+3}(x) + \sum_{n}^{n+2} F_{n}(x)$ | - | $\mathbb{R}$ | 5-term recurrence |
| [Pentanacci](src/arnold/layers/core/polynomial/sequences/pentanacci.py)    | $F_{n+5}(x) = x F_{n+4}(x) + \sum_{n}^{n+3} F_{n}(x)$ | - | $\mathbb{R}$ | 6-term recurrence |
| [Hexanacci](src/arnold/layers/core/polynomial/sequences/hexanacci.py)      | $F_{n+6}(x) = x F_{n+5}(x) + \sum_{n}^{n+4} F_{n}(x)$ | - | $\mathbb{R}$ | 7-term recurrence |
| [Heptanacci](src/arnold/layers/core/polynomial/sequences/heptanacci.py)    | $F_{n+7}(x) = x F_{n+6}(x) + \sum_{n}^{n+5} F_{n}(x)$ | - | $\mathbb{R}$ | 8-term recurrence |
| [Octanacci](src/arnold/layers/core/polynomial/sequences/octanacci.py)      | $F_{n+8}(x) = x F_{n+7}(x) + \sum_{n}^{n+6} F_{n}(x)$ | - | $\mathbb{R}$ | 9-term recurrence |


### Radial basis functions

Using Radial Basis Functions (RBFs) in Kolmogorov-Arnold Networks (KANs) introduces a versatile approach to function approximation that relies on localized basis functions centered around certain points in the input space. RBFs are particularly effective in capturing complex nonlinear relationships.

| Layer                                                                                     | Definition | Parameters | Support | Implementation |
| :-                                                                                        | :- | :- | :- | :- |
| [GaussianRBF](src/arnold/layers/core/rbf/gaussian.py)                             | $\phi(r) = e^{-\varepsilon^2 r^2}$ | $\varepsilon$ | $\mathbb{R}$ | Vectorized | 
| [MultiquadricRBF](src/arnold/layers/core/rbf/multiquadric.py)                     | $\phi(r) = \sqrt{1 + \varepsilon^2 r^2}$ | $\varepsilon$ | $\mathbb{R}$ | Vectorized |
| [InverseMultiQuadricRBF](src/arnold/layers/core/rbf/inverse_multiquadric.py)     | $\phi(r) = 1/\sqrt{1 + \varepsilon^2 r^2}$ | $\varepsilon$ | $\mathbb{R}$ | Vectorized |
| [InverseQuadricRBF](src/arnold/layers/core/rbf/inverse_quadric.py)           | $\phi(r) = 1/(1 + \varepsilon^2 r^2)$ | $\varepsilon$ | $\mathbb{R}$ | Vectorized |
| [ThinPlateSplineRBF](src/arnold/layers/core/rbf/thin_plate_spline.py)             | $\phi(r) = r^2 \ln(r)$ | - | $\mathbb{R}$ | Vectorized |
| [CauchyRBF](src/arnold/layers/core/rbf/cauchy.py)                                 | $\phi(r) = 1/(1 + (r/\sigma)^2)$ | $\sigma$ | $\mathbb{R}$ | Vectorized |
| [LinearRBF](src/arnold/layers/core/rbf/linear.py)                                 | $\phi(r) = r$ | - | $\mathbb{R}$ | Vectorized |
| [CubicRBF](src/arnold/layers/core/rbf/cubic.py)                                   | $\phi(r) = r^3$ | - | $\mathbb{R}$ | Vectorized |
| [PowerRBF](src/arnold/layers/core/rbf/power.py)                                   | $\phi(r) = r^k$ | $k$ | $\mathbb{R}$ | Vectorized |
| [ExponentialRBF](src/arnold/layers/core/rbf/exponential.py)                       | $\phi(r) = e^{-r/\sigma}$ | $\sigma$ | $\mathbb{R}$ | Vectorized | 


### Wavelets

Wavelets in Kolmogorov-Arnold Networks (KANs) offer a sophisticated approach to function approximation by utilizing wavelet basis functions, which are well-suited for capturing both local and global features of complex signals or data distributions.

| Layer                                                             | Definition | Parameters | Support | Implementation |
| :-                                                                | :- | :- | :- | :- |
| [Bump](src/arnold/layers/core/wavelets/bump.py)                         | Smooth compactly-supported | - | $[-1, 1]$ | Vectorized | 
| [DerivativeOfGaussian](src/arnold/layers/core/wavelets/derivative_of_gaussian.py)       | DOG wavelet | $n$ | $\mathbb{R}$ | Vectorized | 
| [Meyer](src/arnold/layers/core/wavelets/meyer.py)                       | Frequency-domain defined | - | $\mathbb{R}$ | FFT |  
| [Morlet](src/arnold/layers/core/wavelets/morelet.py)           | $\psi(t) = e^{i\omega_0 t} e^{-t^2/2}$ | $\omega_0$ | $\mathbb{R}$ | Vectorized |  
| [Poisson](src/arnold/layers/core/wavelets/poisson.py)                   | ${\psi (t)={\frac {1}{\pi }}{\frac {1-t^{2}}{(1+t^{2})^{2}}}}$ | - | $\mathbb{R}$ | Vectorized |  
| [Ricker](src/arnold/layers/core/wavelets/ricker.py)       | ${\psi (t)={\frac {2}{{\sqrt {3\sigma }}\pi ^{1/4}}}\left(1-\left({\frac {t}{\sigma }}\right)^{2}\right)e^{-{\frac {t^{2}}{2\sigma ^{2}}}}}$ | $\sigma$ | $\mathbb{R}$ | Vectorized | 
| [Shannon](src/arnold/layers/core/wavelets/shannon.py)                   | $\psi^{(Sha)}(t)=\mathop{\mathrm{sinc}} \left({\frac {t}{2}}\right)\cdot \cos \left({\frac {3\pi t}{2}}\right)$               | -        | $\mathbb{R}$ | Vectorized | 
| [Haar](src/arnold/layers/core/wavelets/haar.py)                         | Step function wavelet | - | $[0, 1]$ | Vectorized |
| [Daubechies](src/arnold/layers/core/wavelets/daubechies.py)             | $db_N$ wavelets | $N$ | $\mathbb{R}$ | Filter bank |
| [Symlet](src/arnold/layers/core/wavelets/symlet.py)                     | $sym_N$ symmetric wavelets | $N$ | $\mathbb{R}$ | Filter bank |
| [Coiflet](src/arnold/layers/core/wavelets/coiflet.py)                   | $coif_N$ wavelets | $N$ | $\mathbb{R}$ | Filter bank | 


### Spectral bases

Trigonometric and random Fourier basis functions for periodic signals and kernel approximation.

| Layer | Definition | Parameters | Support | Implementation |
| :- | :- | :- | :- | :- |
| [FourierKAN](src/arnold/layers/core/spectral/fourier_basis.py) | $\phi_k(x) = \{1, \cos(k\omega x), \sin(k\omega x)\}$ | $\omega$ | $\mathbb{R}$ | Vectorized trig |
| [RandomFourierFeatures](src/arnold/layers/core/spectral/random_fourier_features.py) | $\phi(x) = \sqrt{2/D} \cos(\omega^T x + b)$ | $\sigma$ | $\mathbb{R}$ | Random sampling |


### Geometric bases

Basis functions for spherical and geometric domains.

| Layer | Definition | Parameters | Support | Implementation |
| :- | :- | :- | :- | :- |
| [Zernike](src/arnold/layers/core/geometric/zernike.py) | $R_n^0(\rho) = P_n(2\rho^2 - 1)$ | - | $[0, 1]$ | Legendre recurrence |
| [SphericalHarmonics](src/arnold/layers/core/geometric/spherical_harmonics.py) | $Y_l^0(\theta) = N_l^0 P_l(\cos\theta)$ | - | $S^2$ | Legendre recurrence |
| [HypersphericalHarmonics](src/arnold/layers/core/geometric/hyperspherical_harmonics.py) | $H_l^{(n)}(\cos\theta) = C_l^{(n-2)/2}(\cos\theta)$ | $n$ | $S^{n-1}$ | Gegenbauer recurrence |


### Special functions

Classical special functions as basis for physics-informed networks.

| Layer | Definition | Parameters | Support | Implementation |
| :- | :- | :- | :- | :- |
| [Airy](src/arnold/layers/core/special/airy.py) | $\text{Ai}(x), \text{Bi}(x)$ | - | $\mathbb{R}$ | Power series |
| [BesselFunctions](src/arnold/layers/core/special/bessel_functions.py) | $J_\nu(x)$ | $\nu$ | $\mathbb{R}$ | Power series |


### Symbolic tools (NEW!)

Extract explicit mathematical formulas from trained KANs — a key advantage for interpretability:

```python
from arnold.layers import Legendre
from arnold.layers.symbolic import kan_to_polynomial, kan_to_latex

# Create and train a layer
layer = Legendre(units=1, degree=3)
layer.build((None, 2))
# ... train the layer ...

# Convert to symbolic expression
expr = kan_to_polynomial(layer)
print(expr)  # e.g., 0.5*x_0**2 - 0.3*x_1 + 1.2*x_0*x_1

# Get LaTeX for papers
latex = kan_to_latex(layer, mode='equation')
```

Requires: `pip install sympy`


### Training utilities

$\mathtt{ARNOLD}$ provides specialized constraints, regularizers, and initializers for KAN layers:

#### Constraints

| Class | Description |
| :- | :- |
| [SoftplusLowerBound](src/arnold/layers/constraints/bounds.py) | Enforce parameter lower bounds via softplus |
| [SigmoidInterval](src/arnold/layers/constraints/bounds.py) | Constrain parameters to an interval via sigmoid |
| [Positivity](src/arnold/layers/constraints/positivity.py) | Ensure strictly positive parameters |
| [Monotonicity](src/arnold/layers/constraints/monotonicity.py) | Enforce monotonic weight sequences |
| [Orthogonality](src/arnold/layers/constraints/orthogonality.py) | Maintain orthogonal weight matrices |

#### Regularizers

| Class | Description |
| :- | :- |
| [L1Regularizer](src/arnold/layers/regularizers/l1_l2.py) | L1 sparsity penalty |
| [L2Regularizer](src/arnold/layers/regularizers/l1_l2.py) | L2 weight decay |
| [L1L2Regularizer](src/arnold/layers/regularizers/l1_l2.py) | Elastic net (combined L1+L2) |
| [SparsityRegularizer](src/arnold/layers/regularizers/sparsity.py) | KL-divergence based sparsity |
| [SmoothnessRegularizer](src/arnold/layers/regularizers/smoothness.py) | Finite-difference smoothness |
| [CurvatureRegularizer](src/arnold/layers/regularizers/curvature.py) | Second-derivative penalty |

#### Initializers

| Class | Description |
| :- | :- |
| [PolynomialInitializer](src/arnold/layers/initializers/polynomial.py) | Degree-scaled coefficient initialization |
| [RBFInitializer](src/arnold/layers/initializers/rbf.py) | Center/width placement strategies |
| [SpectralInitializer](src/arnold/layers/initializers/spectral.py) | Power-law/exponential decay |
| [OrthogonalInitializer](src/arnold/layers/initializers/orthogonal_init.py) | QR-based orthonormal matrices |


### Advanced architectures

Pre-built KAN architecture variants for different use cases:

| Architecture | Description |
| :- | :- |
| [OriginalKAN](src/arnold/layers/architectures/original_kan.py) | B-spline KAN from Liu et al. (2024) |
| [CompactKAN](src/arnold/layers/architectures/ckan.py) | Efficient KAN with configurable basis + residual |
| [KalmanKAN](src/arnold/layers/architectures/kalman_kan.py) | Recursive filter-based KAN for sequences |
| [MLPBasis](src/arnold/layers/architectures/mlp_basis.py) | MLP as learnable basis function |
| [HyperKAN](src/arnold/layers/architectures/hyper_kan.py) | Hypernetwork-based dynamic weight generation |


### Mixed basis layers

Combine multiple basis functions in a single layer:

| Layer | Description |
| :- | :- |
| [MixedBasis](src/arnold/layers/mixed/mixed_basis.py) | Learned softmax-weighted combination |
| [ProductBasis](src/arnold/layers/mixed/product_basis.py) | Tensor product expansions |
| [AttentionBasis](src/arnold/layers/mixed/attention_basis.py) | Attention-weighted dynamic combination |


## Project structure

```
src/arnold/layers/
├── core/
│   ├── common/           # Types, evaluation, parameters, numerics
│   ├── polynomial/
│   │   ├── orthogonal/   # Jacobi, Chebyshev, Legendre, Hermite, etc.
│   │   ├── q_orthogonal/ # q-Hahn, q-Racah, q-Hermite, etc.
│   │   └── sequences/    # Fibonacci, Lucas, Pell, Fermat, etc.
│   ├── spectral/         # Fourier, RandomFourierFeatures
│   ├── geometric/        # Zernike, SphericalHarmonics, Hyperspherical
│   ├── special/          # Airy, Bessel, Mathieu, etc.
│   ├── rbf/              # Gaussian, Multiquadric, ThinPlate, etc.
│   ├── wavelets/         # Haar, Daubechies, Morlet, etc.
│   ├── splines/          # B-spline, Catmull-Rom, Cardinal
│   └── registry.py       # Centralized layer registry (130+ entries)
├── constraints/          # SoftplusLowerBound, Monotonicity, etc.
├── regularizers/         # L1, L2, Sparsity, Smoothness, etc.
├── initializers/         # Polynomial, RBF, Spectral, Orthogonal
├── mixed/                # MixedBasis, ProductBasis, AttentionBasis
├── architectures/        # OriginalKAN, CompactKAN, KalmanKAN, etc.
└── symbolic/             # kan_to_polynomial, kan_to_latex, simplification
```


Developer notes
---------------
- Numerical constants (``PARAM_EPS``, dtype-aware eps) live in ``arnold.utils.constants`` to keep domain clamps consistent.
- Use ``arnold.utils.compilation.kan_function`` to wrap basis evaluators with the standard ``tf.function`` settings; disable ``jit_compile`` there if a kernel is not XLA-friendly.
- Use the **Layer Registry** for string-based layer instantiation: `get_layer("legendre", degree=5, units=32)`
- All layers support Keras serialization via `get_config()` for model saving/loading.
