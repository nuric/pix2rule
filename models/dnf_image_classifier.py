"""Rule learning model for relsgame dataset."""
from typing import Dict, Any, List
import tensorflow as tf
import tensorflow.keras.layers as L

import configlib
from configlib import config as C
from reportlib import ReportLayer

from components.util_layers import (
    MergeFacts,
    SpacialFlatten,
    SpacialBroadcast,
    RecombineStackedImage,
)
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
add_argument = configlib.add_group(
    "DNF Image Model Options", prefix="dnf_image_classifier"
)
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
    add_argument, components.dnf_layer.configurable, prefix="hidden"
)
configlib.add_arguments_dict(
    add_argument, components.dnf_layer.configurable, prefix="inference"
)
add_argument(
    "--iterations",
    type=int,
    default=1,
    help="Number of inference steps to perform.",
)
# ---------------------------


def process_image(image: tf.Tensor, _: Dict[str, Any]) -> tf.Tensor:
    """Process given image input to extract facts."""
    # image (B, W, H, C)
    # ---------------------------
    # Process the images
    image_layer = utils.factory.get_and_init(
        components.inputlayers.image,
        C,
        "dnf_image_classifier_image_",
        name="image_layer",
    )
    raw_objects = image_layer(image)  # (B, W, H, E)
    raw_objects = SpacialFlatten()(raw_objects)  # (B, O, E)
    # raw_objects = L.LayerNormalization()(raw_objects)  # (B, O, E)
    # ---------------------------
    # Apply point wise transformations to the objects
    raw_objects = L.Dense(32, activation="tanh")(raw_objects)
    # raw_objects = L.LayerNormalization()(raw_objects)  # (B, O, E)
    # raw_objects = L.Dense(32, activation="tanh")(raw_objects)
    # ---------------------------
    # Select a subset of objects
    obj_selector = utils.factory.get_and_init(
        components.object_selection,
        C,
        "dnf_image_classifier_object_sel_",
        name="object_selector",
    )
    selected_objects = obj_selector(raw_objects)
    # {'object_scores': (B, N), 'object_atts': (B, N, O), 'objects': (B, N, E)}
    selected_objects = ReportLayer()(selected_objects)
    # ---------------------------
    return selected_objects["objects"]


def process_task_id(task_id: tf.Tensor, input_desc: Dict[str, Any]) -> Dict[str, Any]:
    """Process given task ids."""
    encoded = OneHotCategoricalInput(input_desc["num_categories"], activation="tanh")(
        task_id
    )  # (B, T)
    return {"nullary": encoded}  # facts


def decode_objects_to_image(objects: tf.Tensor) -> tf.Tensor:
    """Decode objects into a full image."""
    # objects (B, O, E)
    decoder_initial_res = [3, 3]
    hidden_size = 32
    decoded = SpacialBroadcast(decoder_initial_res)(objects)  # (B*O, W, H, E)
    decoder_cnn = tf.keras.Sequential(
        [
            L.Conv2DTranspose(
                hidden_size, 5, strides=2, padding="SAME", activation="relu"
            ),
            L.Conv2DTranspose(
                hidden_size, 5, strides=2, padding="SAME", activation="relu"
            ),
            # L.Conv2DTranspose(
            #     hidden_size, 5, strides=1, padding="SAME", activation="relu"
            # ),
            L.Conv2DTranspose(4, 5, strides=1, padding="SAME", activation=None),
        ],
        name="decoder_cnn",
    )
    decoded = decoder_cnn(decoded)  # (B*S, W, H, 4)
    recon_dict = RecombineStackedImage(num_channels=3)(
        [objects, decoded]
    )  # (B, W, H, C)
    recon_dict = ReportLayer(name="recon")(recon_dict)
    return recon_dict["combined"]


def predict_labels_from_facts(
    facts: Dict[str, tf.Tensor], task_description: Dict[str, Any]
) -> tf.Tensor:
    """Use DNF layers to predict the label from given facts."""
    # These are the target rules we want to learn
    target_rules = task_description["outputs"]["label"][
        "target_rules"
    ]  # List of rule arities
    # ---------------------------
    # Check for hidden DNF layer:
    if C["dnf_image_classifier_hidden_arities"]:
        hidden_dnf = utils.factory.get_and_init(
            components.dnf_layer,
            C,
            "dnf_image_classifier_hidden_",
            name="hidden_dnf_layer",
        )
        facts["apply_activation"] = True
        facts_kernel = hidden_dnf(facts)
        facts_kernel = ReportLayer(name="hidden_facts")(facts_kernel)
        facts = {k: facts_kernel[k] for k in ["nullary", "unary", "binary"]}
    # ---------------------------
    # Perform final inference
    dnf_layer = utils.factory.get_and_init(
        components.dnf_layer,
        C,
        "dnf_image_classifier_inference_",
        arities=C["dnf_image_classifier_inference_arities"] + target_rules,
        recursive=C["dnf_image_classifier_iterations"] > 1,
        name="dnf_layer",
    )
    if C["dnf_image_classifier_iterations"] > 1:
        facts = dnf_layer.pad_inputs(facts)  # {'nullary': (B, P0+R0), ...}
    for i in range(C["dnf_image_classifier_iterations"]):
        facts["apply_activation"] = i < (C["dnf_image_classifier_iterations"] - 1)
        facts_kernel = dnf_layer(facts)  # {'nullary': (B, P0+R0), ...}
        facts_kernel = ReportLayer(name="facts" + str(i))(facts_kernel)
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
    return predictions


def build_model(  # pylint: disable=too-many-locals
    task_description: Dict[str, Any]
) -> Dict[str, Any]:
    """Build the DNF trainable model."""
    # ---------------------------
    # Setup and process inputs
    processors = {
        "image": process_image,
        "task_id": process_task_id,
    }
    dnf_inputs = utils.factory.create_input_layers(task_description, processors)
    loss: Dict[str, tf.keras.losses.Loss] = dict()
    metrics: Dict[str, tf.keras.metrics.Metric] = dict()
    outputs: Dict[str, tf.Tensor] = dict()
    # ---------------------------
    # Decode image back
    selected_objects = dnf_inputs["processed"]["image"]  # (B, O, E)
    decoded_image = decode_objects_to_image(selected_objects)  # (B, H, W, C)
    if "image" in task_description["outputs"]:
        outputs["image"] = decoded_image
        loss["image"] = tf.keras.losses.MeanSquaredError()
        metrics["image"] = tf.keras.metrics.MeanAbsoluteError(name="mae")
    # ---------------------------
    # Extract unary and binary features of objects
    facts_list: List[Dict[str, tf.Tensor]] = list()
    object_feat = utils.factory.get_and_init(
        components.object_features,
        C,
        "dnf_image_classifier_object_feat_",
        name="object_features",
    )
    object_facts = object_feat(selected_objects)
    # [{'unary': (B, N, P1), 'binary': (B, N, N-1, P2)}]
    facts_list.append(object_facts)
    if "task_id" in dnf_inputs["processed"]:
        facts_list.append(dnf_inputs["processed"]["task_id"])
    # ---------------------------
    # Merge all the facts
    facts: Dict[str, tf.Tensor] = MergeFacts()(facts_list)
    # {'nullary': (B, P0), 'unary': (B, N, P1), 'binary': (B, N, N-1, P2)}
    facts = ReportLayer()(facts)
    # ---------------------------
    # Perform rule learning and get predictions
    if "label" in task_description["outputs"]:
        predictions = predict_labels_from_facts(facts, task_description)  # (B, R)
        outputs["label"] = predictions
        dataset_type = task_description["outputs"]["label"]["type"]
        dname = task_description["name"]
        lname = C["dnf_image_classifier_inference_layer_name"]
        # ---
        if lname == "DNF":
            expected_type = "multilabel"
            loss["label"] = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            metrics["label"] = [tf.keras.metrics.CategoricalAccuracy(name="acc")]
            if dname == "gendnf":
                metrics["label"] = [tf.keras.metrics.BinaryAccuracy(name="acc")]
        else:
            expected_type = "multiclass"
            loss["label"] = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            )
            metrics["label"] = [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
        assert (
            dataset_type == expected_type
        ), f"DNF requires a {expected_type} dataset, got {dataset_type}"
    # ---
    # ---
    model = tf.keras.Model(
        inputs=dnf_inputs["input_layers"],
        outputs=outputs,
        name="relsgame_model",
    )
    # ---------------------------
    # Setup temperature scheduler callback
    callbacks = [
        utils.callbacks.ParamScheduler(
            layer_params=[("object_selector", "temperature")],
            scheduler=utils.schedules.DelayedExponentialDecay(
                C["dnf_image_classifier_object_sel_initial_temperature"],
                decay_steps=1,
                decay_rate=0.9,
                delay=20,
            ),
            min_max_values=(0.01, 1.0),
        ),
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
                0.05, decay_steps=1, decay_rate=1.07, delay=40
            ),
            min_max_values=(0.0, 6.0),
        ),
    ]
    # ---
    return {"model": model, "loss": loss, "metrics": metrics, "callbacks": callbacks}
