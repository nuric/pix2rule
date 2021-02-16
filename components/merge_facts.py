"""Layers to merge input facts together."""
from typing import List, Dict
import tensorflow as tf


class MergeFacts(tf.keras.layers.Layer):
    """Merge dictionary based fact tensors."""

    def call(self, inputs: List[Dict[str, tf.Tensor]], **kwargs):
        """Merging facts based on arity by concatenating them."""
        # inputs [{'nullary': ..., 'binary': ...}, {'binary': ...}]
        # This layer needs to be used with care as it assumes that every feature of every
        # object is already computed and we just need to concatenate them. That is, all facts
        # contain all objects and their features are concatenated.
        assert len(inputs) > 1, f"Nothing to merge, got {len(inputs)} inputs."
        facts = inputs[0]  # Start with left most and reduce towards right
        for fact_dict in inputs[1:]:
            for key, tensor in fact_dict.items():
                if key in facts:
                    # We need to merge
                    facts[key] = tf.concat([facts[key], tensor], -1)
                else:
                    # Add into the facts list directly
                    facts[key] = tensor
        return facts
