"""Models library for custom layers and models."""
import components
from . import rule_learner

# We expose a list of custom layers for saving and loading models
custom_layers = {l.__name__: l for l in [rule_learner.RuleLearner]}
# Merge into custom component layers
custom_layers.update(components.custom_layers)
