"""Reusable components used in the project."""

from . import sequence_features
from . import relsgame_cnn

# We expose a list of custom layers for saving and loading models
custom_layers = [sequence_features.SequenceFeatures, relsgame_cnn.RelationsGameCNN]
custom_layers = {l.__name__: l for l in custom_layers}
