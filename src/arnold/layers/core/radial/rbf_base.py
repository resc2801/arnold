from abc import abstractmethod
import tensorflow as tf

from arnold.layers.core.kan_base import KANBase

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="RBFBase")
class RBFBase(KANBase):
    """
    Abstract base class for Kolmogorov-Arnold Network layer using radial basis functions.
    """

    def __init__(
            self, 
            *args,
            grid_min:float=-1.0, 
            grid_max:float=1.0, 
            num_grids:int=8, 
            spline_weight_init_scale:float=0.1,
            **kwargs
        ):
        """
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

        :param spline_weight_init_scale: Factor to initialize the spline_weights, i.e. N(0, 1.0 * self.spline_weight_init_scale)
        :type spline_weight_init_scale: float

        :param tanh_x: Flag indicating whether to normalize any input to [-1, 1] using tanh before further processing.
        :type tanh_x: bool
        """
        super().__init__(*args, **kwargs)

        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.spline_weight_init_scale = spline_weight_init_scale

        self.grid = tf.constant(
            tf.linspace(self.grid_min, self.grid_max, self.num_grids),
        )

        self.spline_weight = self.add_weight(
            shape=(self.input_dim * self.num_grids, self.output_dim),
            initializer=tfk.initializers.RandomNormal(
                mean=0.0, 
                stddev=1.0 * self.spline_weight_init_scale
            ),
            trainable=True
        )
    
    @tf.function
    def call(self, inputs):
        # Normalize x to [-1, 1] using tanh
        x = tf.tanh(inputs) if self.tanh_x else inputs
        x = tf.reshape(x, (-1, self.input_dim, 1))

        y = tf.matmul(
            self.get_basis(x),
            self.spline_weight
        )

        return y
    
    @abstractmethod
    def get_basis(self, x):
        """
        Computes the basis for given `x`.

        :param x: Data to compute the basis with.
        :type x: tf.Tensor

        :returns: basis
        :rtype: tf.Tensor
        """
        raise NotImplementedError(
            f"Layer {self.__class__.__name__} does not have a "
            "`get_basis()` method implemented."
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "grid_min": self.grid_min,
            "grid_max": self.grid_max,
            "num_grids": self.num_grids,
            "spline_weight_init_scale": self.spline_weight_init_scale,
        })
        return config
