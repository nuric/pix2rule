"""Global reporting tool to extract eager tensors."""
from typing import Dict, Any
import pdb

import tensorflow as tf

# Global report dictionary
_report: Dict[str, Any] = dict()


# Models / layers use this function to write into the global dictionary
def report_tensor(key: str, tensor: tf.Tensor):
    """Report a tensor to collect at debug or predict time."""
    assert isinstance(key, str), "Keys must be strings."
    key_count = sum([1 for k in _report if k.startswith(key)])
    if key_count:
        key = key + str(key_count)
    _report[key] = tensor


def create_report(model: tf.keras.Model, dataset: tf.data.Dataset) -> Dict[str, Any]:
    """Take 1 batch from dataset and perform a forward pass of model."""
    _report.clear()
    _report["debug"] = True
    for inputs, outputs in dataset.take(1):
        _report.update({"in_" + k: v for k, v in inputs.items()})
        _report.update({"out_" + k: v for k, v in outputs.items()})
        # Models should populate global report dictionary
        model_outs = model(inputs, training=False)
        _report.update({"prediction_" + k: v for k, v in model_outs.items()})
        # which we then collate here
    _report["debug"] = False
    return {k: v.numpy() for k, v in _report.items() if k != "debug"}


def report_break():
    """Setup reporting breakpoint."""
    if _report.get("debug", False):
        pdb.set_trace()


class ReportLayer(tf.keras.layers.Layer):
    """Reporting layer for logging tensors."""

    def call(self, inputs: Dict[str, tf.Tensor], **kwargs):
        """Report and return every tensor passing through."""
        for key, tensor in inputs.items():
            report_tensor(key, tensor)
        return inputs
