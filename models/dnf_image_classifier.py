"""Rule learning model for relsgame dataset."""
from typing import Dict, Any
import tensorflow as tf
import tensorflow.keras.layers as L

from reportlib import ReportLayer
from components.util_layers import MergeFacts
from components.util_layers import SpacialFlatten
from components.inputlayers.categorical import OneHotCategoricalInput
from components.inputlayers.image import RelationsGameImageInput
from components.object_features import LinearObjectFeatures
from components.object_selection import RelaxedObjectSelection
from .dnf_layer import DNFLayer

# import configlib
# from configlib import config as C


def build_model(
    data_description: Dict[str, Any]
) -> tf.keras.Model:  # pylint: disable=too-many-locals
    """Build the trainable model."""
    # ---------------------------
    # Setup inputs
    image = L.Input(shape=(12, 12, 3), name="image", dtype="float32")  # (B, W, H, C)
    task_id = L.Input(shape=(), name="task_id", dtype="int32")  # (B,)
    task_facts = OneHotCategoricalInput(
        data_description["inputs"]["task_id"]["num_categories"]
    )(task_id)
    # {'nullary': (B, T)}
    # ---------------------------
    # Process the images
    raw_objects = RelationsGameImageInput()(image)  # (B, W, H, E)
    raw_objects = SpacialFlatten()(raw_objects)  # (B, O, E)
    # raw_objects = RelationsGamePixelCNN()(image)  # (B, W*H, E)
    # raw_objects = Shuffle()(raw_objects)  # (B, O, E)
    # ---------------------------
    # Select a subset of objects
    obj_selector = RelaxedObjectSelection()
    # obj_selector = SlotAttention(
    # num_iterations=3, num_slots=3, slot_size=32, mlp_hidden_size=32
    # )
    selected_objects = obj_selector(raw_objects)
    # {'object_scores': (B, N), 'object_atts': (B, N, O), 'objects': (B, N, E)}
    selected_objects = ReportLayer()(selected_objects)
    # ---------------------------
    # Extract unary and binary features
    facts = LinearObjectFeatures()(selected_objects["objects"])
    # {'unary': (B, N, P1), 'binary': (B, N, N-1, P2)}
    facts = MergeFacts()([task_facts, facts])
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
        inputs=[image, task_id],
        outputs=predictions,
        name="relsgame_model",
    )
    # ---------------------------
    # Compile model for training
    dataset_type = data_description["output"]["type"]
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
