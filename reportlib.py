"""Global reporting tool to extract eager tensors."""
from typing import Dict, Any
import pdb

import tensorflow as tf

report: Dict[str, Any] = dict()


def create_report(model: tf.keras.Model, dataset: tf.data.Dataset) -> Dict[str, Any]:
    """Take 1 batch from dataset and perform a forward pass of model."""
    report.clear()
    report["debug"] = True
    for inputs, label in dataset.take(1):
        # Models should populate global report dictionary
        report.update(inputs)
        report["label"] = label
        report["output"] = model(inputs, training=False)
    report["debug"] = False
    return {k: v.numpy() for k, v in report.items() if k != "debug"}


def report_break():
    """Setup reporting breakpoint."""
    if report.get("debug", False):
        pdb.set_trace()
