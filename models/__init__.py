"""Models library for custom layers and models."""
from typing import Dict, Any
import tensorflow as tf

import components
import configlib
from configlib import config as C

from . import rule_learner
from . import sequences_model

from . import relsgame_model

# ---------------------------
# We expose a list of custom layers for saving and loading models
custom_layers: Dict[str, type] = {
    l.__name__: l
    for l in [rule_learner.SequencesRuleLearner, rule_learner.BaseRuleLearner]
}
# Merge into custom component layers
custom_layers.update(components.custom_layers)
# ---------------------------

# ---------------------------
# Model registry
# registry = {m.__name__.split(".")[-1]: m.build_model for m in [sequence_model]}
# type checker seems to not recognise what is going above
registry = {
    "image_classifier": relsgame_model.build_model,
}
# ---------------------------

# ---------------------------
# Model configuration / selection
parser = configlib.add_parser("Global model options.")
parser.add_argument(
    "--model_name",
    default="sequences_model",
    choices=registry.keys(),
    help="Model name to train / evaluate.",
)
# ---------------------------


def build_model(data_description: Dict[str, Any], name: str = None) -> tf.keras.Model:
    """Build given model by name."""
    return registry[name or C["model_name"]](data_description)
