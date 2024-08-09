from abc import ABC
import tensorflow as tf

tfk = tf.keras
tfkl = tfk.layers


@tfk.utils.register_keras_serializable(package="arnold", name="KANBase")
class KANBase(tfkl.Layer, ABC):
    r"""
    Abstract base class for Kolmogorov-Arnold Network layers.
    """

    def __init__(
            self, 
            input_dim:int, 
            output_dim:int, 
            *args,
            tanh_x:bool=True,
            **kwargs):
        r"""
        :param input_dim: This layers input size
        :type input_dim: int

        :param output_dim: This layers output size
        :type output_dim: int
        
        :param tanh_x: Flag indicating whether to normalize any input to [-1, 1] using tanh before further processing.
        :type tanh_x: bool
        """
        super().__init__(**kwargs)

        self.input_dim = input_dim 
        self.output_dim = output_dim
        self.tanh_x = tanh_x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        r"""
        Returns the config of the object. 
        An object config is a Python dictionary (serializable) containing the information needed to re-instantiate it. 

        :returns: A Python dictionary containing the configuration of this object.
        :rtype: dict
        """
        base_config = super().get_config()
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "tanh_x": self.tanh_x,
        }
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        r"""
        Creates a layer from its config.
        This method is the reverse of get_config, capable of instantiating the same layer from the config dictionary. 
        It does not handle layer connectivity (handled by Network), nor weights (handled by set_weights).
        
        :param config: A Python dictionary, typically the output of get_config.
        :type config: dict
        
        :returns: A layer instance.
        :rtype: KANBase
        """
        return cls(**config)