from abc import abstractmethod
import tensorflow as tf

from arnold.layers.core.kan_base import KANBase

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="RBFBase")
class RBFBase(KANBase):
    r"""
    Abstract base class for Kolmogorov-Arnold Network layer using radial basis functions.

    TODO:
        - Wendland RBF 
        - Cauchy RBF
    """

    def __init__(
            self, 
            *args,
            grid_min:float=0.0, 
            grid_max:float=1.0, 
            num_grids:int=8, 
            **kwargs
        ):
        r"""
        :param input_dim: This layers input size
        :type input_dim: int

        :param output_dim: This layers output size
        :type output_dim: int

        :param grid_min: Specifies the lower grid range, defaults to 0.
        :type grid_min: float

        :param grid_max: Specifies the upper grid range, defaults to 1.
        :type grid_max: float

        :param num_grids: Specifies the number of equidistant grid points.
        :type num_grids: int

        :param tanh_x: Flag indicating whether to normalize any input to [-1, 1] using tanh before further processing.
        :type tanh_x: bool
        """
        super().__init__(*args, **kwargs)

        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids    

        self.grid = tf.constant(
            tf.linspace(self.grid_min, self.grid_max, self.num_grids),
        )

        self.kernel_weights = self.add_weight(
            shape=(self.input_dim, self.num_grids, self.output_dim),
            # shape=(self.input_dim * self.num_grids, self.output_dim),
            initializer=tfk.initializers.RandomNormal(),
            trainable=True
        )

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def radii(self, x):
        r"""
        Computes :math:`\lVert x  - x_i \rVert` for equidistant :math:`x_i \in [\text{grid_min}, \text{grid_max}]`

        :param x: Data points to compute the radii with.
        :type inputs: tf.Tensor

        :returns: :math:`\lVert x  - x_i \rVert`
        :rtype: tf.Tensor
        """
        return tf.math.abs((x - self.grid) / ((self.grid_max - self.grid_min) / (self.num_grids - 1)))
    
    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def call(self, inputs):
        # Normalize x to [-1, 1] using tanh
        x = tf.tanh(inputs) if self.tanh_x else inputs
        x = tf.reshape(x, (-1, self.input_dim, 1))

        y = tf.einsum(
            'bid,ido->bo', 
            self.get_kernels(self.radii(x)), 
            self.kernel_weights,
            optimize='auto'
        )
        # y = tf.matmul(
        #     self.get_kernels(self.radii(x)),
        #     self.kernel_weights
        # )

        return y
    
    @abstractmethod
    def get_kernels(self, r):
        r"""
        Evaluates the radial basis kernels for given :math:`r = \lVert x - x_{i} \rVert`.

        :param r: Data to compute the basis with.
        :type r: tf.Tensor

        :returns: Evaluated radial kernels
        :rtype: tf.Tensor
        """
        raise NotImplementedError(
            f"Layer {self.__class__.__name__} does not have a "
            "`get_kernels()` method implemented."
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "grid_min": self.grid_min,
            "grid_max": self.grid_max,
            "num_grids": self.num_grids,
            "kernel_weight_init": self.kernel_weight_init,
        })
        return config


class ExponentialRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the Exponential radial basis function

    :math:`\phi(r) = e^{ - \frac{r}{\sigma}}, \quad \sigma \in \mathbb{R}, r = \lVert x - x_{i} \rVert`
    """

    def __init__(self, 
                 *args,
                 sigma_init: float | None = None, 
                 sigma_trainable=True,
                 **kwargs):
        r"""
        :param input_dim: This layers input size
        :type input_dim: int

        :param output_dim: This layers output size
        :type output_dim: int

        :param grid_min: Specifies the lower grid range, defaults to -1.
        :type grid_min: float

        :param grid_max: Specifies the upper grid range, defaults to -1.
        :type grid_max: float

        :param num_grids: Specifies the grid range.
        :type num_grids: int

        :param sigma_init: Initial value for shape parameter :math:`sigma`. Defaults to None (a initialized to RandomNormal).
        :type sigma_init: float

        :param sigma_trainable: Flag indicating whether :math:`sigma` is a trainable parameter. Defaults to True
        :type sigma_trainable: bool

        :param tanh_x: Flag indicating whether to normalize any input to [-1, 1] using tanh before further processing.
        :type tanh_x: bool
        """
        super().__init__(*args, **kwargs)

        self.sigma_init = sigma_init
        self.sigma_trainable = sigma_trainable

        self.sigma_logits = self.add_weight(
            shape=(1,self.input_dim,1),
            initializer=tfk.initializers.Constant(value=tf.math.log(self.sigma_init)) if self.sigma_init else tfk.initializers.RandomNormal(),
            trainable=self.sigma_trainable,
            name='sigma_logits'
        )

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def get_kernels(self, r):
        return tf.math.exp(-(r / tf.math.exp(self.sigma_logits)))

    def get_config(self):
        config = super().get_config()
        config.update({
            "sigma_init": self.sigma_init,
            "sigma_trainable": self.sigma_trainable,
        })
        return config


class PowerRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the Power radial basis function

    :math:`\phi(r) = r^{p}, \quad p \in \mathbb{R}, r = \lVert x - x_{i} \rVert`
    """

    def __init__(self, 
                 *args,
                 power_init: float | None = None, 
                 power_trainable=True,
                 **kwargs):
        r"""
        :param input_dim: This layers input size
        :type input_dim: int

        :param output_dim: This layers output size
        :type output_dim: int

        :param grid_min: Specifies the lower grid range, defaults to -1.
        :type grid_min: float

        :param grid_max: Specifies the upper grid range, defaults to -1.
        :type grid_max: float

        :param num_grids: Specifies the grid range.
        :type num_grids: int

        :param power_init: Initial value for the exponents. Defaults to None (a initialized to RandomNormal).
        :type epsilon_init: float

        :param power_trainable: Flag indicating whether `power` is a trainable parameter. Defaults to True
        :type power_trainable: bool

        :param tanh_x: Flag indicating whether to normalize any input to [-1, 1] using tanh before further processing.
        :type tanh_x: bool
        """
        super().__init__(*args, **kwargs)

        self.power_init = power_init
        self.power_trainable = power_trainable

        self.power_logits = self.add_weight(
            shape=(1,self.input_dim,1),
            initializer=tfk.initializers.Constant(value=tf.math.log(self.power_init)) if self.power_init else tfk.initializers.RandomNormal(),
            trainable=self.power_trainable,
            name='power_logits'
        )

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def get_kernels(self, r):
        return tf.math.pow(r, tf.math.exp(self.power_logits))

    def get_config(self):
        config = super().get_config()
        config.update({
            "power_init": self.power_init,
            "power_trainable": self.power_trainable,
        })
        return config
    

class ThinPlateSplineRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the Thin plate spline radial basis function

    :math:`\phi(r) = r^{2} \ln(r), \quad r = \lVert x - x_{i} \rVert`
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def get_kernels(self, r):
        r"""
        :math:`\phi(r) = r^{2} \ln(r) = r \ln(r^{r})`
        """
        return tf.math.log(tf.math.pow(r,r)) * r


class GaussianRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the Gaussian radial basis function

    :math:`\phi(r) = e^{-(\epsilon r)^{2}}, \quad r = \lVert x - x_{i} \rVert`

    with shape-parameter tuning per input dimension.
    """

    def __init__(self, 
                 *args,
                 epsilon_init:float=None, 
                 epsilon_trainable=True,
                 **kwargs):
        r"""
        :param input_dim: This layers input size
        :type input_dim: int

        :param output_dim: This layers output size
        :type output_dim: int

        :param grid_min: Specifies the lower grid range, defaults to -1.
        :type grid_min: float

        :param grid_max: Specifies the upper grid range, defaults to -1.
        :type grid_max: float

        :param num_grids: Specifies the grid range.
        :type num_grids: int

        :param epsilon_init: Initial value for the non-negative shape parameter epsilon. Defaults to None.
        :type epsilon_init: float

        :param epsilon_trainable: Flag indicating whether shape parameter epsilon is a trainable parameter. Defaults to True.
        :type epsilon_trainable: bool

        :param tanh_x: Flag indicating whether to normalize any input to [-1, 1] using tanh before further processing.
        :type tanh_x: bool
        """
        super().__init__(*args, **kwargs)

        self.epsilon_init = epsilon_init
        self.epsilon_trainable = epsilon_trainable

        self.epsilon_logits = self.add_weight(
            shape=(1,self.input_dim,1),
            initializer=tfk.initializers.Constant(value=tf.math.log(self.epsilon_init)) if self.epsilon_init else tfk.initializers.RandomNormal(),
            trainable=self.epsilon_trainable,
            name='epsilon_logits'
        )

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def get_kernels(self, r):
        return tf.exp(- (tf.exp(self.epsilon_logits)*r)**2)

    def get_config(self):
        config = super().get_config()
        config.update({
            "epsilon_init": self.epsilon_init,
            "epsilon_trainable": self.epsilon_trainable,
        })
        return config


class CubicRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the Cubic radial basis function

    :math:`\phi(r) = r^{3}, \quad r = \lVert x - x_{i} \rVert`
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def get_kernels(self, r):
        return tf.math.pow(r, 3)


class LinearRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the Linear radial basis function

    :math:`\phi(r) = r, \quad r = \lVert x - x_{i} \rVert`
    """

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def get_kernels(self, r):
        return r


class InverseQuadricRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using a inverse quadric radial basis function

    :math:`\phi(r) = \frac{1}(1+(\epsilon r)^{2}}, \quad r = \lVert x - x_{i} \rVert`
    """

    def __init__(self, 
                 *args,
                 epsilon_init:float=None, 
                 epsilon_trainable=True,
                 **kwargs):
        r"""
        :param input_dim: This layers input size
        :type input_dim: int

        :param output_dim: This layers output size
        :type output_dim: int

        :param grid_min: Specifies the lower grid range, defaults to -1.
        :type grid_min: float

        :param grid_max: Specifies the upper grid range, defaults to -1.
        :type grid_max: float

        :param num_grids: Specifies the grid range.
        :type num_grids: int

        :param epsilon_init: Initial value for the non-negative shape parameter epsilon. Defaults to None.
        :type epsilon_init: float

        :param epsilon_trainable: Flag indicating whether shape parameter epsilon is a trainable parameter. Defaults to True.
        :type epsilon_trainable: bool

        :param tanh_x: Flag indicating whether to normalize any input to [-1, 1] using tanh before further processing.
        :type tanh_x: bool
        """
        super().__init__(*args, **kwargs)

        self.epsilon_init = epsilon_init
        self.epsilon_trainable = epsilon_trainable

        self.epsilon_logits = self.add_weight(
            shape=(1,self.input_dim,1),
            initializer=tfk.initializers.Constant(value=tf.math.log(self.epsilon_init)) if self.epsilon_init else tfk.initializers.RandomNormal(),
            trainable=self.epsilon_trainable,
            name='epsilon_logits'
        )

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def get_kernels(self, r):
        return tf.math.divide(1.0, (1.0 + (tf.exp(self.epsilon_logits)*r)**2))

    def get_config(self):
        config = super().get_config()
        config.update({
            "epsilon_init": self.epsilon_init,
            "epsilon_trainable": self.epsilon_trainable,
        })
        return config
    

class MultiquadricRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the Multiquadric radial basis function

    :math:`\phi(r) = \sqrt{1+(\epsilon r)^{2}}, \quad r = \lVert x - x_{i} \rVert`

    with shape-parameter tuning per input dimension.
    """

    def __init__(self, 
                 *args,
                 epsilon_init:float=None, 
                 epsilon_trainable=True,
                 **kwargs):
        r"""
        :param input_dim: This layers input size
        :type input_dim: int

        :param output_dim: This layers output size
        :type output_dim: int

        :param grid_min: Specifies the lower grid range, defaults to -1.
        :type grid_min: float

        :param grid_max: Specifies the upper grid range, defaults to -1.
        :type grid_max: float

        :param num_grids: Specifies the grid range.
        :type num_grids: int

        :param epsilon_init: Initial value for the non-negative shape parameter epsilon. Defaults to None.
        :type epsilon_init: float

        :param epsilon_trainable: Flag indicating whether shape parameter epsilon is a trainable parameter. Defaults to True.
        :type epsilon_trainable: bool

        :param tanh_x: Flag indicating whether to normalize any input to [-1, 1] using tanh before further processing.
        :type tanh_x: bool
        """
        super().__init__(*args, **kwargs)

        self.epsilon_init = epsilon_init
        self.epsilon_trainable = epsilon_trainable

        self.epsilon_logits = self.add_weight(
            shape=(1,self.input_dim,1),
            initializer=tfk.initializers.Constant(value=tf.math.log(self.epsilon_init)) if self.epsilon_init else tfk.initializers.RandomNormal(),
            trainable=self.epsilon_trainable,
            name='epsilon_logits'
        )

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def get_kernels(self, r):
        return tf.math.sqrt(1.0 + (tf.exp(self.epsilon_logits)*r)**2)

    def get_config(self):
        config = super().get_config()
        config.update({
            "epsilon_init": self.epsilon_init,
            "epsilon_trainable": self.epsilon_trainable,
        })
        return config


class InverseMultiQuadricRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the inverse multiquadric radial basis function

    :math:`\phi(r) = \frac{1}(\sqrt{1+(\epsilon r)^{2}}}, \quad r = \lVert x - x_{i} \rVert`

    with shape-parameter tuning per input dimension.
    """

    def __init__(self, 
                 *args,
                 epsilon_init:float=None, 
                 epsilon_trainable=True,
                 **kwargs):
        r"""
        :param input_dim: This layers input size
        :type input_dim: int

        :param output_dim: This layers output size
        :type output_dim: int

        :param grid_min: Specifies the lower grid range, defaults to -1.
        :type grid_min: float

        :param grid_max: Specifies the upper grid range, defaults to -1.
        :type grid_max: float

        :param num_grids: Specifies the grid range.
        :type num_grids: int

        :param epsilon_init: Initial value for the non-negative shape parameter epsilon. Defaults to None.
        :type epsilon_init: float

        :param epsilon_trainable: Flag indicating whether shape parameter epsilon is a trainable parameter. Defaults to True.
        :type epsilon_trainable: bool

        :param tanh_x: Flag indicating whether to normalize any input to [-1, 1] using tanh before further processing.
        :type tanh_x: bool
        """
        super().__init__(*args, **kwargs)

        self.epsilon_init = epsilon_init
        self.epsilon_trainable = epsilon_trainable

        self.epsilon_logits = self.add_weight(
            shape=(1,self.input_dim,1),
            initializer=tfk.initializers.Constant(value=tf.math.log(self.epsilon_init)) if self.epsilon_init else tfk.initializers.RandomNormal(),
            trainable=self.epsilon_trainable,
            name='epsilon_logits'
        )

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def get_kernels(self, r):
        return tf.math.divide(1.0, tf.math.sqrt(1.0 + (tf.exp(self.epsilon_logits)*r)**2))

    def get_config(self):
        config = super().get_config()
        config.update({
            "epsilon_init": self.epsilon_init,
            "epsilon_trainable": self.epsilon_trainable,
        })
        return config


class CauchyRBF(RBFBase):
    r"""
    Kolmogorov-Arnold Network layer using the Cauchy radial basis function

    :math:`\phi(r) = \frac{1}{1 + (\frac{r}{\sigma})^{2}}, \quad \sigma \in \mathbb{R}, r = \lVert x - x_{i} \rVert`
    """

    def __init__(self, 
                 *args,
                 sigma_init: float | None = None, 
                 sigma_trainable=True,
                 **kwargs):
        r"""
        :param input_dim: This layers input size
        :type input_dim: int

        :param output_dim: This layers output size
        :type output_dim: int

        :param grid_min: Specifies the lower grid range, defaults to -1.
        :type grid_min: float

        :param grid_max: Specifies the upper grid range, defaults to -1.
        :type grid_max: float

        :param num_grids: Specifies the grid range.
        :type num_grids: int

        :param sigma_init: Initial value for shape parameter :math:`sigma`. Defaults to None (a initialized to RandomNormal).
        :type sigma_init: float

        :param sigma_trainable: Flag indicating whether :math:`sigma` is a trainable parameter. Defaults to True
        :type sigma_trainable: bool

        :param tanh_x: Flag indicating whether to normalize any input to [-1, 1] using tanh before further processing.
        :type tanh_x: bool
        """
        super().__init__(*args, **kwargs)

        self.sigma_init = sigma_init
        self.sigma_trainable = sigma_trainable

        self.sigma_logits = self.add_weight(
            shape=(1,self.input_dim,1),
            initializer=tfk.initializers.Constant(value=tf.math.log(self.sigma_init)) if self.sigma_init else tfk.initializers.RandomNormal(),
            trainable=self.sigma_trainable,
            name='sigma_logits'
        )

    @tf.function(autograph=True, jit_compile=True, reduce_retracing=True, experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def get_kernels(self, r):
        return tf.math.divide(
            1.0,
            1.0 + tf.math.square(tf.math.divide(r, tf.math.exp(self.sigma_logits)))
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "sigma_init": self.sigma_init,
            "sigma_trainable": self.sigma_trainable,
        })
        return config
    
