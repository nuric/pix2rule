"""Rule learning model for DNF dataset."""
from typing import Dict, Any, List
import tensorflow as tf

import configlib
from configlib import config as C
from reportlib import ReportLayer

from components.util_layers import MergeFacts
import components.dnf_layer

import utils.callbacks
import utils.factory
import utils.schedules


# ---------------------------
# Setup configurable parameters of the model
add_argument = configlib.add_group("DNF Rule Model Options", prefix="dnf_rule_learner")
# ---
# DNF Layer options
configlib.add_arguments_dict(
    add_argument, components.dnf_layer.configurable, prefix="inference"
)
# ---------------------------


def build_model(  # pylint: disable=too-many-locals
    task_description: Dict[str, Any]
) -> Dict[str, Any]:
    """Build the DNF trainable model."""
    # ---------------------------
    # Setup and process inputs
    # Pass through processors for already encoded data
    processors = {
        "nullary": lambda x, _: {"nullary": x},
        "unary": lambda x, _: {"unary": x},
        "binary": lambda x, _: {"binary": x},
    }
    dnf_inputs = utils.factory.create_input_layers(task_description, processors)
    facts_list: List[Dict[str, tf.Tensor]] = list(dnf_inputs["processed"].values())
    # ---------------------------
    # Merge all the facts
    facts: Dict[str, tf.Tensor] = MergeFacts()(facts_list)
    # {'nullary': (B, P0), 'unary': (B, N, P1), 'binary': (B, N, N-1, P2)}
    facts = ReportLayer(name="facts0")(facts)
    # ---------------------------
    # Perform rule learning and get predictions
    target_rules = task_description["outputs"]["label"][
        "target_rules"
    ]  # List of rule arities
    dnf_layer = utils.factory.get_and_init(
        components.dnf_layer,
        C,
        "dnf_rule_learner_inference_",
        arities=target_rules,
        num_conjuncts=task_description["metadata"]["num_conjuncts"],
        num_total_variables=task_description["metadata"]["num_variables"],
        name="dnf_layer",
    )
    facts_kernel = dnf_layer(facts)  # {'nullary': (B, P0+R0), ...}
    facts_kernel = ReportLayer(name="facts1")(facts_kernel)
    # ---
    # Extract out the required target rules
    predictions = facts_kernel["nullary"]  # (B, 1)
    # ---------------------------
    # Create model with given inputs and outputs
    loss: Dict[str, tf.keras.losses.Loss] = dict()
    metrics: Dict[str, tf.keras.metrics.Metric] = {
        "label": [tf.keras.metrics.BinaryAccuracy(name="acc")]
    }
    outputs: Dict[str, tf.Tensor] = {"label": predictions}
    # ---
    if C["dnf_rule_learner_inference_layer_name"] == "DNF":
        loss["label"] = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    else:
        loss["label"] = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # ---
    model = tf.keras.Model(
        inputs=dnf_inputs["input_layers"],
        outputs=outputs,
        name="dnf_rule_learner",
    )
    # ---------------------------
    # Setup temperature scheduler callback
    callbacks = [
        # utils.callbacks.ParamScheduler(
        #     layer_params=[("dnf_layer", "temperature")],
        #     scheduler=utils.schedules.DelayedExponentialDecay(
        #         1.0, decay_steps=1, decay_rate=0.8, delay=100
        #     ),
        #     min_max_values=(0.01, 1.0),
        # ),
        utils.callbacks.ParamScheduler(
            layer_params=[("dnf_layer", "success_threshold")],
            scheduler=utils.schedules.DelayedExponentialDecay(
                0.2, decay_steps=2, decay_rate=1.1, delay=10
            ),
            min_max_values=(0.0, 6.0),
        ),
    ]
    # ---
    return {"model": model, "loss": loss, "metrics": metrics, "callbacks": callbacks}
