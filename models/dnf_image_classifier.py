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
from components.dnf_layer import DNFLayer

import utils.callbacks
import utils.factory


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
add_argument(
    "--hidden_predicates",
    type=int,
    nargs="*",
    default=[],
    help="Hidden extra predicates to be learned.",
)
add_argument(
    "--num_total_variables", type=int, default=2, help="Number of variables in the DNF."
)
add_argument(
    "--num_conjuncts", type=int, default=8, help="Number of conjunctions in the DNF."
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
    dnf_layer = DNFLayer(
        arities=target_rules + C["dnf_hidden_predicates"],
        num_total_variables=C["dnf_num_total_variables"],
        num_conjuncts=C["dnf_num_conjuncts"],
        recursive=True,
    )
    padded_facts = dnf_layer.pad_inputs(facts)  # {'nullary': (B, P0+R0), ...}
    for _ in range(C["dnf_iterations"]):
        padded_facts_kernel = dnf_layer(padded_facts)  # {'nullary': (B, P0+R0), ...}
        padded_facts_kernel = ReportLayer()(padded_facts_kernel)
        padded_facts = {
            k: padded_facts_kernel[k] for k in ["nullary", "unary", "binary"]
        }
    # ---
    # Extract out the required target rules
    predictions = list()
    for arity, key in enumerate(["nullary", "unary", "binary"]):
        count = target_rules.count(arity)
        if count:
            predictions.append(padded_facts[key][..., -count:])  # (B, ..., Rcount)
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
    assert (
        dataset_type == "multilabel"
    ), f"DNF classifier requires a multilabel dataset, got {dataset_type}"
    # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    metrics = [tf.keras.metrics.CategoricalAccuracy(name="acc")]
    # metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    # ---------------------------
    # Setup temperature scheduler callback
    callbacks = [
        utils.callbacks.ParamScheduler(
            layer_params=[("object_selector", "temperature")],
            scheduler=tf.keras.optimizers.schedules.ExponentialDecay(
                0.5, decay_steps=1, decay_rate=0.9
            ),
            min_value=0.01,
        ),
        utils.callbacks.ParamScheduler(
            layer_params=[("dnf_layer", "temperature")],
            scheduler=tf.keras.optimizers.schedules.ExponentialDecay(
                1.0, decay_steps=2, decay_rate=0.9
            ),
            min_value=0.01,
        ),
    ]
    # ---
    return {"model": model, "loss": loss, "metrics": metrics, "callbacks": callbacks}
