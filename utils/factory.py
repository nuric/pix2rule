"""Factory modules to initialise hyper-parameters of layers."""
from typing import Dict, Any
import tensorflow as tf


def get_and_init(
    module, config: Dict[str, Any], prefix: str, **kwargs
) -> tf.keras.layers.Layer:
    """Get and initialise a layer with a given prefix based on configuration."""
    layer_class = getattr(module, config[prefix + "layer_name"])
    # Gather remaining arguments
    config_args = {
        argname[len(prefix) :]: value
        for argname, value in config.items()
        if argname.startswith(prefix) and not argname.endswith("layer_name")
    }
    config_args.update(kwargs)
    return layer_class(**config_args)
