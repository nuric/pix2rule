"""Rule learning model for relsgame dataset."""
from typing import Dict, Any
import tensorflow as tf

import configlib
from configlib import config as C
from reportlib import ReportLayer

from components.util_layers import MergeFacts, SpacialFlatten
from components.ops import flatten_concat
from components.inputlayers.categorical import OneHotCategoricalInput
import components.inputlayers.image
import components.object_features
import components.object_selection
import components.dnf_layer

import utils.callbacks
import utils.factory
import utils.schedules


# ---------------------------
# Setup configurable parameters of the model
add_argument = configlib.add_group("DNF Image Model Options", prefix="dnf")
# ---
# Image layer parameters
configlib.add_arguments_dict(
    add_argument, components.inputlayers.image.configurable, prefix="image"
)
# ---
# Object selection
configlib.add_arguments_dict(
    add_argument, components.object_selection.configurable, prefix="object_sel"
)
# ---
# Object features
configlib.add_arguments_dict(
    add_argument, components.object_features.configurable, prefix="object_feat"
)
# ---
# DNF Layer options
configlib.add_arguments_dict(
    add_argument, components.dnf_layer.configurable, prefix="inference"
)
add_argument(
    "--iterations",
    type=int,
    default=2,
    help="Number of inference steps to perform.",
)
# ---------------------------


def process_image(image: tf.Tensor, _: Dict[str, Any]) -> Dict[str, tf.Tensor]:
    """Process given image input to extract facts."""
    # image (B, W, H, C)
    # ---------------------------
    # Process the images
    image_layer = utils.factory.get_and_init(
        components.inputlayers.image, C, "dnf_image_", name="image_layer"
    )
    raw_objects = image_layer(image)  # (B, W, H, E)
    raw_objects = SpacialFlatten()(raw_objects)  # (B, O, E)
    # ---------------------------
    # Select a subset of objects
    obj_selector = utils.factory.get_and_init(
        components.object_selection, C, "dnf_object_sel_", name="object_selector"
    )
    selected_objects = obj_selector(raw_objects)
    # {'object_scores': (B, N), 'object_atts': (B, N, O), 'objects': (B, N, E)}
    selected_objects = ReportLayer()(selected_objects)
    # ---------------------------
    # Extract unary and binary features
    object_feat = utils.factory.get_and_init(
        components.object_features, C, "dnf_object_feat_", name="object_features"
    )
    facts: Dict[str, tf.Tensor] = object_feat(selected_objects["objects"])
    # {'unary': (B, N, P1), 'binary': (B, N, N-1, P2)}
    return facts


def process_task_id(task_id: tf.Tensor, input_desc: Dict[str, Any]) -> Dict[str, Any]:
    """Process given task ids."""
    encoded = OneHotCategoricalInput(input_desc["num_categories"])(task_id)  # (B, T)
    return {"nullary": encoded}  # facts


def build_model(  # pylint: disable=too-many-locals
    task_description: Dict[str, Any]
) -> Dict[str, Any]:
    """Build the DNF trainable model."""
    # ---------------------------
    # Setup and process inputs
    processors = {"image": process_image, "task_id": process_task_id}
    dnf_inputs = utils.factory.create_input_layers(task_description, processors)
    # ---------------------------
    facts = MergeFacts()(list(dnf_inputs["processed"].values()))
    # {'nullary': (B, P0), 'unary': (B, N, P1), 'binary': (B, N, N-1, P2)}
    facts = ReportLayer()(facts)
    # ---------------------------
    # Perform rule learning and get predictions
    target_rules = task_description["output"]["target_rules"]  # List of rule arities
    # predictions = tf.keras.layers.Dense(len(target_rules))(
    #     flatten_concat(list(facts.values()))
    # )
    dnf_layer = utils.factory.get_and_init(
        components.dnf_layer,
        C,
        "dnf_inference_",
        arities=target_rules,
        recursive=C["dnf_iterations"] > 1,
        name="dnf_layer",
    )
    if C["dnf_iterations"] > 1:
        facts = dnf_layer.pad_inputs(facts)  # {'nullary': (B, P0+R0), ...}
    for _ in range(C["dnf_iterations"]):
        facts_kernel = dnf_layer(facts)  # {'nullary': (B, P0+R0), ...}
        facts_kernel = ReportLayer()(facts_kernel)
        facts = {k: facts_kernel[k] for k in ["nullary", "unary", "binary"]}
    # ---
    # Extract out the required target rules
    predictions = list()
    for arity, key in enumerate(["nullary", "unary", "binary"]):
        count = target_rules.count(arity)
        if count:
            predictions.append(facts[key][..., -count:])  # (B, ..., Rcount)
    predictions = flatten_concat(predictions)  # (B, R)
    # ---------------------------
    # Create model with given inputs and outputs
    model = tf.keras.Model(
        inputs=dnf_inputs["input_layers"],
        outputs=predictions,
        name="relsgame_model",
    )
    # ---------------------------
    # Compile model for training
    dataset_type = task_description["output"]["type"]
    if C["dnf_inference_layer_name"] == "DNF":
        expected_type = "multilabel"
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        metrics = [tf.keras.metrics.CategoricalAccuracy(name="acc")]
    elif C["dnf_inference_layer_name"] == "RealDNF":
        expected_type = "multiclass"
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
    assert (
        dataset_type == expected_type
    ), f"DNF requires a {expected_type} dataset, got {dataset_type}"
    # ---------------------------
    # Setup temperature scheduler callback
    callbacks = [
        utils.callbacks.ParamScheduler(
            layer_params=[("object_selector", "temperature")],
            scheduler=utils.schedules.DelayedExponentialDecay(
                C["dnf_object_sel_initial_temperature"],
                decay_steps=1,
                decay_rate=0.9,
                delay=40,
            ),
            min_value=0.01,
        ),
        utils.callbacks.ParamScheduler(
            layer_params=[("dnf_layer", "temperature")],
            scheduler=utils.schedules.DelayedExponentialDecay(
                1.0, decay_steps=1, decay_rate=0.8, delay=100
            ),
            min_value=0.01,
        ),
    ]
    # ---
    return {"model": model, "loss": loss, "metrics": metrics, "callbacks": callbacks}
