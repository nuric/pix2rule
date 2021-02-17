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
from components.object_selection import RelaxedObjectSelection

from .dnf_layer import DNFLayer


parser = configlib.add_parser("DNF Image Model Options.")
configlib.add_arguments_dict(
    parser, components.inputlayers.image.configurable, prefix="--dnf_image_"
)
parser.add_argument(
    "--dnf_img_noise",
    action="store_true",
    help="Optional add noise to image input before processing.",
)


def process_image(image: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Process given image input to extract facts."""
    # image (B, W, H, C)
    # ---------------------------
    # Process the images
    image_layer = getattr(components.inputlayers.image, C["dnf_image_layer_name"])
    image_layer = image_layer(
        hidden_size=C["dnf_image_hidden_size"], activation=C["dnf_image_activation"]
    )
    raw_objects = image_layer(image)  # (B, W, H, E)
    raw_objects = SpacialFlatten()(raw_objects)  # (B, O, E)
    # ---------------------------
    # Select a subset of objects
    obj_selector = RelaxedObjectSelection()
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
) -> tf.keras.Model:  # pylint: disable=too-many-locals
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
    return model
