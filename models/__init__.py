"""Models library for custom layers and models."""
from typing import Dict
import tensorflow as tf

import components
import configlib
from configlib import config as C

from . import rule_learner
from . import sequence_model


# We expose a list of custom layers for saving and loading models
custom_layers: Dict[str, type] = {l.__name__: l for l in [rule_learner.RuleLearner]}
# Merge into custom component layers
custom_layers.update(components.custom_layers)

# Model registry
# registry = {m.__name__.split(".")[-1]: m.build_model for m in [sequence_model]}
# type checker seems to not recognise what is going above
registry = {"sequence_model": sequence_model.build_model}

# Model configuration / selection
parser = configlib.add_parser("Global model options.")
parser.add_argument(
    "--model_name",
    default="sequence_model",
    choices=registry.keys(),
    help="Model name to train / evaluate.",
)


def build_model(name: str = None) -> tf.keras.Model:
    """Build given model by name."""
    return registry[name or C["model_name"]]()
