"""Models library for custom layers and models."""
from typing import Dict, Any
import tensorflow as tf

import configlib
from configlib import config as C

from . import dnf_image_classifier
from . import predinet
from . import mlp_image_classifier
from . import slot_autoencoder

# ---------------------------
# Model registry
# registry = {m.__name__.split(".")[-1]: m.build_model for m in [sequence_model]}
# type checker seems to not recognise what is going above
registry = {
    "dnf_image_classifier": dnf_image_classifier.build_model,
    "predinet": predinet.build_model,
    "slot_ae": slot_autoencoder.build_model,
    "mlp_image_classifier": mlp_image_classifier.build_model,
}
# ---------------------------

# ---------------------------
# Model configuration / selection
add_argument = configlib.add_group("Global model options", prefix="model")
add_argument(
    "--name",
    default=next(iter(registry.keys())),
    choices=registry.keys(),
    help="Model name to train / evaluate.",
)
# ---------------------------


def build_model(data_description: Dict[str, Any], name: str = None) -> tf.keras.Model:
    """Build given model by name."""
    return registry[name or C["model_name"]](data_description)
