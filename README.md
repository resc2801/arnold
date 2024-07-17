# ![alt text](arnold.png) $\mathtt{ARNOLD}$ 


$\mathtt{ARNOLD}$ offers `tf.keras` implementations of [Kolmogorov-Arnold Networks (KAN) layers](https://arxiv.org/pdf/2404.19756?trk=public_post_main-feed-card-text) using various basis functions.


Kolmogorov-Arnold Networks (KANs) are a type of neural network inspired by a mathematical theorem related to the representation of continuous functions. 

The Kolmogorov-Arnold representation theorem states that any multivariate continuous function can be represented as a superposition of continuous functions of a single variable and addition. Leveraging this theorem, KANs are designed to approximate complex multivariate functions by breaking them down into simpler, univariate functions.



## Installation
Install $\mathtt{ARNOLD}$ via pip

```shell
python3 pip -m install arnold
```

## Usage 
Simply use $\mathtt{ARNOLD}$'s KAN layers as a drop-in-replacement for `tf.keras.layers.Dense` and mix with any standard layers.

```python
fancy_kan =tfk.Sequential([
        tfkl.Reshape(target_shape=(2, )),
        tfkl.Rescaling(scale=1./127.5, offset=-1),
        Chebyshev1st(input_dim=2, output_dim=8, degree=2),
        tfkl.LayerNormalization(),
        Legendre(input_dim=8, output_dim=6, degree=3),
        tfkl.LayerNormalization(),
        Bump(input_dim=6, output_dim=1),
        tfkl.Activation(tfk.activations.sigmoid)
    ],
    name="fancy_kan" 
)
```

## Examples

| Task | Dataset |  |
| :- | :- | :-: |
| Multinomial Classifcation             | MNIST         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/resc2801/arnold/blob/main/examples/mnist/mnist.ipynb)   |
| Binary classification                 | "Two moons"   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/resc2801/arnold/blob/main/examples/two_moons/two_moons.ipynb)   |
| Multivariate Function Interpolation   | 2D function   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/resc2801/arnold/blob/main/examples/multivariate_interpolation/multivariate_interpolation.ipynb)   |

## Available KAN Layers

### Polynomial bases

In the context of KANs, polynomial bases can be employed to represent the univariate functions that constitute the network. When incorporating polynomial bases into KANs, the network's layers are designed to transform the input variables into polynomial functions. These polynomial functions serve as the univariate components specified by the Kolmogorov-Arnold representation theorem. By using polynomials, the network can efficiently approximate smooth, continuous functions, exploiting the well-known properties of polynomials such as their ability to be easily differentiated and integrated.

#### Continuous orthogonal polynomials

Orthogonal polynomials, such as Legendre, Chebyshev, and Hermite polynomials, have coefficients that are uncorrelated when integrated over a certain range with a specific weight function.
The orthogonality condition ensures that each polynomial basis function captures unique aspects of the data, minimizing overlap and improving the network's learning efficiency.

| Layer                                                                                             | Definition | Parameters | Support | Implementation |
| :-                                                                                                | :- | :- | :- | :- | 
| [Al-Salam-Carlitz (1st kind)](src/arnold/layers/polynomial/orthogonal/al_salam_carlitz.py)        | $`U^{(a)}_{n+1} (x;q) = (x - (1 + a) q^{n}) U^{(a)}_{n} (x;q) + a q^{n-1} (1 - q^{n}) U^{(a)}_{n-1} (x;q)`$ | $a, q$ | $\mathbb{R}$ | three-term recurence |
| [Al-Salam-Carlitz (2nd kind)](src/arnold/layers/polynomial/orthogonal/al_salam_carlitz.py)        | $`V^{a}_{n+1} (x; q) = U^{a}_{n+1} (x; \frac{1}{q})`$ | $a, q$ | $\mathbb{R}$ | three-term recurence |
| [Askey-Wilson](src/arnold/layers/polynomial/orthogonal/askey_wilson.py)                           | $`p_{n}(x;a,b,c,d\mid q) = a^{-n}(ab,ac,ad;q)_{n} \; {}_{4}{\phi}_{3} \left[\begin{matrix}q^{-n}&abcdq^{n-1}&ae^{i\theta }&ae^{-i\theta }\\ab&ac&ad\end{matrix};q,q\right]`$ | $a, b, c, d$ | $\mathbb{R}$ | three-term recurence |
| [Bannai-Ito](src/arnold/layers/polynomial/orthogonal/bannai_ito.py)                               | $`y_{n}(x)=\sum_{k=0}^{n}{\frac {(n+k)!}{(n-k)!k!}}\,\left({\frac {x}{2}}\right)^{k}`$ |- | $\mathbb{R}$ | 3-term recurrence |
| [Bessel](src/arnold/layers/polynomial/orthogonal/bessel.py)                                       | $`y_{n}(x)=\sum_{k=0}^{n}{\frac {(n+k)!}{(n-k)!k!}}\,\left({\frac {x}{2}}\right)^{k}`$ |- | $\mathbb{R}$ | 3-term recurrence |
| [Chebyshev (1st kind)](src/arnold/layers/polynomial/orthogonal/chebyshev.py)                      | $`T_{n}(\cos \theta )=\cos(n\theta)`$ | - | $\mathbb{R}$ | trigonometric |
| [Chebyshev (2nd kind)](src/arnold/layers/polynomial/orthogonal/chebyshev.py)                      | $`U_{n}(\cos \theta )\sin \theta =\sin ((n+1)\theta), \; \cos(\theta) = \tfrac{2x - (a+b)}{b-a}, \; \theta \in [0, \pi]`$ | $a, b$ | $[a, b]$ | trigonometric |
| [Chebyshev (3rd kind)](src/arnold/layers/polynomial/orthogonal/chebyshev.py)                      | $`V_{n}(x) = \tfrac{\cos(n+1/2) \theta}{\cos(\theta/2)}, \; \cos(\theta) = \tfrac{2x - (a+b)}{b-a}, \; \theta \in [0, \pi]`$ | $a, b$ | $[a, b]$ | trigonometric |
| [Chebyshev (4th kind)](src/arnold/layers/polynomial/orthogonal/chebyshev.py)                      | $`W_{n}(x) = \tfrac{\sin(n+1/2)\theta}{\sin(\theta / 2)}, \; \cos(\theta) = \tfrac{2x - (a+b)}{b-a}, \; \theta \in [0, \pi]`$ | $a, b$ | $[a, b]$ | trigonometric |
| [Gegenbauer](src/arnold/layers/polynomial/orthogonal/gegenbauer.py)                               | $`C_{n}^{\alpha}(x)=\frac {(2\alpha +n-1)!}{(2\alpha -1)! \,n!}\, {}_{2}F_{1}\left(-n,2\alpha +n;\alpha +{\frac {1}{2}}; {\frac {1-x}{2}}\right)`$ | $\alpha > -\tfrac{1}{2}$  | $\mathbb{R}$ | 3-term recurence |
| [Physicist's Hermite](src/arnold/layers/polynomial/orthogonal/hermite.py)                         | $`H_{n}(x)=(-1)^{n}e^{x^{2}}{\frac {d^{n}}{dx^{n}}}e^{-x^{2}}`$ | - | $\mathbb{R}$ | 3-term recurence |
| [Jacobi](src/arnold/layers/polynomial/orthogonal/jacobi.py)                                       | $`P_{n}^{(\alpha ,\beta )}(x)={\frac {(\alpha +1)_{n}}{n!}} \, {}_{2}F_{1}\left(-n,1+\alpha +\beta +n;\alpha +1;{\tfrac {1}{2}}(1-x)\right)`$ | $\alpha, \beta$ | $\mathbb{R}$ | 3-term recurence |
| [Generalized Laguerre](src/arnold/layers/polynomial/orthogonal/laguerre.py)                       | $`L_{n}^{(\alpha )}(x)=\sum _{i=0}^{n}(-1)^{i}{n+\alpha  \choose n-i}{\frac {x^{i}}{i!}}`$ | $\alpha$ | $\mathbb{R}$ | 3-term recurence |
| [Legendre](src/arnold/layers/polynomial/orthogonal/legendre.py)                                   | $`P_{n}(x)=\sum _{k=0}^{\lfloor n/2\rfloor }(-1)^{k}{\frac {(2n-2k)!\ }{(n-k)!\ (n-2k)!\ k!\ 2^{n}}}x^{n-2k}`$ | - | $\mathbb{R}$ | 3-term recurence |
| [Associated Meixner-Pollaczek](src/arnold/layers/polynomial/orthogonal/meixner_pollaczek.py)      | $`(n + c + 1) P^{\lambda}_{n+1}(x; \phi, c) = 2x \sin(\phi) + 2(n + c + \lambda) P^{\lambda}_{n}(x; \phi, c) - (n + c + 2\lambda - 1) P^{\lambda}_{n-1}(x; \phi, c)`$ | $c, \lambda, \phi$ | $\mathbb{R}$ | 3-term recurence |
| [Pollaczek](src/arnold/layers/polynomial/orthogonal/pollaczek.py)                                 | $`n P_{n}(x;a,b) = ((2n-1+2a)x+2b)P_{n-1}(x;a,b)-(n-1)P_{n-2}(x;a,b)`$ | $a, b$ | $\mathbb{R}$ | 3-term recurence |
| Wilson | - | - | - | - | TBD |


#### Discrete orthogonal polynomials

Discrete orthogonal polynomial bases are sequences of polynomials that are pairwise orthogonal with respect to a discrete measure.

| Layer                                                                                             | Definition | Parameters | Support | Implementation |
| :-                                                                                                | :- | :- | :- | :- | 
| [Charlier](src/arnold/layers/polynomial/orthogonal/charlier.py)                                   | $`y_{n}(x)=\sum_{k=0}^{n}{\frac {(n+k)!}{(n-k)!k!}}\,\left({\frac {x}{2}}\right)^{k}`$ | - | $\mathbb{R}$ | 3-term recurrence |
| Discrete Chebyshev    | - | - | - | - |
| Dual Hahn             | - | - | - | - |
| Hahn                  | - | - | - | - |
| Krawtchouk            | - | - | - | - |
| Meixner               | - | - | - | - |
| Racah                 | - | - | - | - |

#### Non-orthogonal polynomials

Non-orthogonal polynomials do not satisfy the condition of orthogonality, meaning their inner product is not necessarily zero for distinct polynomials. 
This may allow for a broader selection of basis functions, which can be tailored to fit particular types of data or specific functional forms that might be more challenging to capture with orthogonal polynomials.

| Layer | Parameters | Support | Examples | Implementation |
| :- | :-: | :-: | :-:   | :- |
| [Lucas](src/arnold/layers/polynomial/w_polynomials/lucas.py)  | - | $\mathbb{R}$ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/resc2801/arnold/blob/main/examples/notebooks/lucas.ipynb) |  3-term recurrence |
| [Laurent](src/arnold/layers/polynomial/laurent.py)            | - | $\mathbb{R}$ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/resc2801/arnold/blob/main/examples/notebooks/lucas.ipynb) |  3-term recurrence |
| Bernstein | - | - | - | - | TBD |

#### Lucas polynomial sequences

A Lucas polynomial sequence is a pair of generalized polynomials which generalize the Lucas sequence to polynomials. We offer a number of special cases.

| Layer                                                                                 | Definition | Parameters | Support | Implementation |
| :-                                                                                    | :- | :- | :- | :- |
| [Chebyshev (1st kind)](src/arnold/layers/polynomial/orthogonal/chebyshev.py)          | $`T_{n}(\cos \theta )=\cos(n\theta)`$ | - | $\mathbb{R}$ | trigonometric |
| [Chebyshev (2nd kind)](src/arnold/layers/polynomial/orthogonal/chebyshev.py)          | $`U_{n}(\cos \theta )\sin \theta =\sin ((n+1)\theta), \; \cos(\theta) = \tfrac{2x - (a+b)}{b-a}, \; \theta \in [0, \pi]`$ | $a, b$ | $[a, b]$ | trigonometric |
| [Fermat](src/arnold/layers/polynomial/w_polynomials/fermat.py)                        | $F_{n+1}(x) = 3x F_{n}(x) - 2 F_{n-1}(x)$   | - | $\mathbb{R}$ | 3-term recurrence |
| [Fermat-Lucas](src/arnold/layers/polynomial/w_polynomials/fermat_lucas.py)            | $f_{n+1}(x) = 3x f_{n}(x) - 2 f_{n-1}(x)$   | - | $\mathbb{R}$ | 3-term recurrence |
| [Fibonacci](src/arnold/layers/polynomial/fibonacci/fibonacci.py)                      | $F_{n+2}(x) = x F_{n+1}(x) + F_{n}(x)$ | - | $\mathbb{R}$ | 3-term recurrence |
| [Jacobsthal](src/arnold/layers/polynomial/w_polynomials/jacobsthal.py)                | $J_{n+1}(x) = 1 J_{n}(x) + 2x J_{n-1}(x)$   | - | $\mathbb{R}$ | 3-term recurrence |
| [Jacobsthal-Lucas](src/arnold/layers/polynomial/w_polynomials/jacobsthal_lucas.py)    | $j_{n+1}(x) = 1 j_{n}(x) + 2x j_{n-1}(x)$   | - | $\mathbb{R}$ | 3-term recurrence |
| [Lucas](src/arnold/layers/polynomial/w_polynomials/lucas.py)                          | $L_{n+1}(x) = x L_{n}(x) + L_{n-1}(x)$           | - | $\mathbb{R}$ | 3-term recurrence |
| [Pell](src/arnold/layers/polynomial/w_polynomials/pell.py)                            | $P_{n+1}(x) = 2 x * P_{n}(x) + P_{n-1}(x)$       | - | $\mathbb{R}$ | 3-term recurrence |
| [Pell-Lucas](src/arnold/layers/polynomial/w_polynomials/pell_lucas.py)                | $Q_{n+1}(x) = x Q_{n}(x) + Q_{n-1}(x)$           | - | $\mathbb{R}$ | 3-term recurrence |


#### (Generalized) Fibbonacci polynomials

The Fibonacci polynomials are a polynomial sequence which can be considered as a generalization of the Fibonacci numbers.
We also provide polynomial sequences based on Fibonacci numbers of higher order. 

| Layer                                                                 | Definition | Parameters | Support | Implementation |
| :-                                                                    | :- | :-: | :-: | :- |
| [Fibonacci](src/arnold/layers/polynomial/fibonacci/fibonacci.py)      | $F_{n+2}(x) = x F_{n+1}(x) + F_{n}(x)$ | - | $\mathbb{R}$ | 3-term recurrence |
| [Tetranacci](src/arnold/layers/polynomial/fibonacci/tetranacci.py)    | $F_{n+4}(x) = x F_{n+3}(x) + \sum_{n}^{n+2} F_{n}(x)$ | - | $\mathbb{R}$ | 5-term recurrence |
| [Pentanacci](src/arnold/layers/polynomial/fibonacci/pentanacci.py)    | $F_{n+5}(x) = x F_{n+4}(x) + \sum_{n}^{n+3} F_{n}(x)$ | - | $\mathbb{R}$ | 6-term recurrence |
| [Hexanacci](src/arnold/layers/polynomial/fibonacci/hexanacci.py)      | $F_{n+6}(x) = x F_{n+5}(x) + \sum_{n}^{n+4} F_{n}(x)$ | - | $\mathbb{R}$ | 7-term recurrence |
| [Heptanacci](src/arnold/layers/polynomial/fibonacci/heptanacci.py)    | $F_{n+7}(x) = x F_{n+6}(x) + \sum_{n}^{n+5} F_{n}(x)$ | - | $\mathbb{R}$ | 8-term recurrence |
| [Octanacci](src/arnold/layers/polynomial/fibonacci/octanacci.py)      | $F_{n+8}(x) = x F_{n+7}(x) + \sum_{n}^{n+6} F_{n}(x)$ | - | $\mathbb{R}$ | 9-term recurrence |


### Radial basis functions

Using Radial Basis Functions (RBFs) in Kolmogorov-Arnold Networks (KANs) introduces a versatile approach to function approximation that relies on localized basis functions centered around certain points in the input space. RBFs are particularly effective in capturing complex nonlinear relationships.

| Layer                                                                                     | Definition | Parameters | Support | Implementation |
| :-                                                                                        | :- | :- | :- | :- |
| [Gaussian RBF](src/arnold/layers/radialbasis/gaussian_rbf.py)                             | - | - | - | - | 
| [Inverse quadratic RBF](src/arnold/layers/radialbasis/inverse_quadratic_rbf.py)           | - | - | - | - | 
| [Inverse multiquadric RBF](src/arnold/layers/radialbasis/inverse_multiquadric_rbf.py)     | - | - | - | - | 


### Wavelets

Wavelets in Kolmogorov-Arnold Networks (KANs) offer a sophisticated approach to function approximation by utilizing wavelet basis functions, which are well-suited for capturing both local and global features of complex signals or data distributions.

| Layer                                                             | Definition | Parameters | Support | Implementation |
| :-                                                                | :- | :- | :- | :- |
| [Bump](src/arnold/layers/wavelet/bump.py)                         | - | - | - | - | 
| [Difference of Gaussians]()                                       | - | - | - | - | TBD |
| [Meyer](src/arnold/layers/wavelet/meyer.py)                       | - | - | - | - |  
| [Morelet (Gabor)](src/arnold/layers/wavelet/morelet.py)           | - | - | - | - |  
| [Poisson](src/arnold/layers/wavelet/poisson.py)                   | ${\psi (t)={\frac {1}{\pi }}{\frac {1-t^{2}}{(1+t^{2})^{2}}}}$ | - | $\mathbb{R}$ | - |  
| [Ricker (Mexican Hat)](src/arnold/layers/wavelet/ricker.py)       | ${\psi (t)={\frac {2}{{\sqrt {3\sigma }}\pi ^{1/4}}}\left(1-\left({\frac {t}{\sigma }}\right)^{2}\right)e^{-{\frac {t^{2}}{2\sigma ^{2}}}}}$ | $\sigma$ | $\mathbb{R}$ |  - | 
| [Shannon](src/arnold/layers/wavelet/shannon.py)                   | - | - | - | - | 

