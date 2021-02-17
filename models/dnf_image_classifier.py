"""Rule learning model for relsgame dataset."""
from typing import Dict, Any
import tensorflow as tf
import tensorflow.keras.layers as L

import configlib
from configlib import config as C
from reportlib import ReportLayer

from components.util_layers import MergeFacts
from components.util_layers import SpacialFlatten
from components.inputlayers.categorical import OneHotCategoricalInput
import components.inputlayers.image
from components.object_features import LinearObjectFeatures
import components.object_selection

import utils.callbacks

from .dnf_layer import DNFLayer


# ---------------------------
# Setup configurable parameters of the model
parser = configlib.add_parser("DNF Image Model Options.")
# ---
# Image layer parameters
parser.add_argument(
    "--dnf_img_noise_stddev",
    type=float,
    default=0.0,
    help="Optional noise to add image input before processing.",
)
configlib.add_arguments_dict(
    parser, components.inputlayers.image.configurable, prefix="--dnf_image_"
)
# ---
# Object selection
configlib.add_arguments_dict(
    parser, components.object_selection.configurable, prefix="--dnf_object_sel_"
)
# ---------------------------


def get_and_init(module, prefix: str, **kwargs) -> tf.keras.layers.Layer:
    """Get and initialise a layer with a given prefix based on configuration."""
    layer_class = getattr(module, C[prefix + "layer_name"])
    config_args = {
        argname[len(prefix) :]: value
        for argname, value in C.items()
        if argname.startswith(prefix) and not argname.endswith("layer_name")
    }
    config_args.update(kwargs)
    return layer_class(**config_args)


def process_image(image: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Process given image input to extract facts."""
    # image (B, W, H, C)
    # ---------------------------
    # Optional noise
    if C["dnf_image_noise_stddev"] > 0:
        image = L.GaussianNoise(C["dnf_image_noise_stddev"])(image)
    # ---------------------------
    # Process the images
    image_layer = get_and_init(
        components.inputlayers.image, "dnf_image_", name="image_layer"
    )
    raw_objects = image_layer(image)  # (B, W, H, E)
    raw_objects = SpacialFlatten()(raw_objects)  # (B, O, E)
    # ---------------------------
    # Select a subset of objects
    obj_selector = get_and_init(
        components.object_selection, "dnf_object_sel_", name="object_selector"
    )
    selected_objects = obj_selector(raw_objects)
    # {'object_scores': (B, N), 'object_atts': (B, N, O), 'objects': (B, N, E)}
    selected_objects = ReportLayer()(selected_objects)
    # ---------------------------
    # Extract unary and binary features
    facts: Dict[str, tf.Tensor] = LinearObjectFeatures()(selected_objects["objects"])
    # {'unary': (B, N, P1), 'binary': (B, N, N-1, P2)}
    return facts


def build_model(
    task_description: Dict[str, Any]
) -> Dict[str, Any]:  # pylint: disable=too-many-locals
    """Build the trainable model."""
    # ---------------------------
    # Setup and process inputs
    input_layers = dict()
    all_facts = list()
    for input_name, input_desc in task_description["inputs"].items():
        input_layer = L.Input(
            shape=input_desc["shape"][1:],
            name=input_name,
            dtype=input_desc["dtype"].name,
        )
        input_layers[input_name] = input_layer
        if input_desc["type"] == "image":
            facts = process_image(input_layer)
            # {'unary': (B, N, P1), 'binary': (B, N, N-1, P2)}
        elif input_desc["type"] == "categorical":
            facts = OneHotCategoricalInput(input_desc["num_categories"])(input_layer)
            # {'nullary': (B, T)}
        all_facts.append(facts)
    # ---------------------------
    facts = MergeFacts()(all_facts)
    # {'nullary': (B, P0), 'unary': (B, N, P1), 'binary': (B, N, N-1, P2)}
    # ---------------------------
    # Perform rule learning and get predictions
    dnf_layer = DNFLayer(arities=[0, 0, 0, 0], recursive=True)
    padded_facts = dnf_layer.pad_inputs(facts)  # {'nullary': (B, P0+R0), ...}
    for _ in range(2):
        padded_facts = dnf_layer(padded_facts)  # {'nullary': (B, P0+R0), ...}
    predictions = padded_facts["nullary"][..., -4:]  # (B, R0)
    # ---------------------------
    # Create model with given inputs and outputs
    model = tf.keras.Model(
        inputs=input_layers,
        outputs=predictions,
        name="relsgame_model",
    )
    # ---------------------------
    # Compile model for training
    dataset_type = task_description["output"]["type"]
    assert (
        dataset_type == "multilabel"
    ), f"DNF classifier requires a multilabel dataset, got {dataset_type}"
    model.compile(
        optimizer="adam",
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        # metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")],
    )
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
    return {"model": model, "callbacks": callbacks}
