"""Base MLP model."""
from typing import Dict, Any
import tensorflow as tf
import tensorflow.keras.layers as L

import configlib
from configlib import config as C

from components.inputlayers.categorical import OneHotCategoricalInput
import components.inputlayers.image

import utils.factory

# Setup configurable parameters of the model
add_argument = configlib.add_group(
    "MLP Image Model Options.", prefix="mlp_image_classifier"
)
# ---
# Image layer parameters
configlib.add_arguments_dict(
    add_argument, components.inputlayers.image.configurable, prefix="image"
)
# ---
# Predinet Layer options
add_argument(
    "--hidden_sizes",
    type=int,
    nargs="+",
    default=[32],
    help="Hidden layer sizes, length determines number of layers.",
)
add_argument(
    "--hidden_activations",
    nargs="+",
    default=["relu"],
    help="Hidden layer activations, must match hidden_sizes.",
)
# ---------------------------


def process_image(image: tf.Tensor, _: Dict[str, Any]) -> tf.Tensor:
    """Process given image input extract objects."""
    # image (B, W, H, C)
    image_layer = utils.factory.get_and_init(
        components.inputlayers.image, C, "mlp_image_", name="image_layer"
    )
    raw_objects = image_layer(image)  # (B, W, H, E)
    return L.Flatten()(raw_objects)  # (B, W*H*E)


def process_task_id(task_id: tf.Tensor, input_desc: Dict[str, Any]) -> tf.Tensor:
    """Process given task ids."""
    return OneHotCategoricalInput(input_desc["num_categories"])(task_id)  # (B, T)


def build_model(  # pylint: disable=too-many-locals
    task_description: Dict[str, Any]
) -> Dict[str, Any]:
    """Build the predinet model."""
    # ---------------------------
    # Setup and process inputs
    processors = {"image": process_image, "task_id": process_task_id}
    mlp_inputs = utils.factory.create_input_layers(task_description, processors)
    # ---------------------------
    # Concatenate processed inputs
    concat_in = next(iter(mlp_inputs["processed"].values()))
    if len(mlp_inputs["processed"]) > 1:
        concat_in = L.Concatenate()(list(mlp_inputs["processed"].values()))
    # ---------------------------
    for size, activation in zip(C["mlp_hidden_sizes"], C["mlp_hidden_activations"]):
        concat_in = L.Dense(size, activation=activation)(concat_in)
    predictions = L.Dense(task_description["output"]["num_categories"])(concat_in)
    # ---------------------------
    # Create model instance
    model = tf.keras.Model(
        inputs=mlp_inputs["input_layers"],
        outputs=predictions,
        name="mlp_image_classifier",
    )
    # ---------------------------
    # Compile model for training
    dataset_type = task_description["output"]["type"]
    assert (
        dataset_type == "binary"
    ), f"MLP image classifier requires a binary classification dataset, got {dataset_type}"
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.keras.metrics.BinaryAccuracy(name="acc")
    # ---------------------------
    return {"model": model, "loss": loss, "metrics": metrics}
