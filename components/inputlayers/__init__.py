"""Input processing layers."""
from typing import Dict
import inspect

import tensorflow as tf

from . import categorical, image, categorical_sequence

# Collect all categorical inputs
registry: Dict[str, tf.keras.layers.Layer] = {
    name: layer
    for module in [categorical, image]
    for name, layer in inspect.getmembers(module)
    if inspect.isclass(layer) and name.endswith("Input")
}
