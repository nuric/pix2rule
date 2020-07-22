"""Models library for custom layers and models."""
from . import rule_learner
from . import seq_feats

# We expose a list of custom layers for saving and loading models
custom_layers = {
    l.__name__: l for l in [seq_feats.SequenceFeatures, rule_learner.RuleLearner]
}
