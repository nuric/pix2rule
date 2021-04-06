"""Factory modules to initialise hyper-parameters of layers."""
from typing import Dict, Any, Callable
import tensorflow as tf


def create_input_layers(
    task_description: Dict[str, Any],
    processors: Dict[str, Callable[[tf.keras.layers.Input, Dict[str, Any]], Any]],
) -> Dict[str, Any]:
    """Create and process input layers based on task description and given processors."""
    processed: Dict[str, Any] = {"input_layers": list(), "processed": dict()}
    for input_name, input_desc in task_description["inputs"].items():
        input_layer = tf.keras.layers.Input(
            shape=input_desc["shape"][1:],
            name=input_name,
            dtype=input_desc["dtype"],
        )
        processed["input_layers"].append(input_layer)
        processed["processed"][input_name] = processors[input_name](
            input_layer, input_desc
        )
    return processed


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
