"""The PrediNet model."""
from typing import Dict, Any
import tensorflow as tf
import tensorflow.keras.layers as L

import configlib
from configlib import config as C
from reportlib import ReportLayer

from components.util_layers import SpacialFlatten
from components.inputlayers.categorical import OneHotCategoricalInput
import components.inputlayers.image

import utils.factory

# Setup configurable parameters of the model
add_argument = configlib.add_group("Predinet Image Model Options.", prefix="predinet")
# ---
# Image layer parameters
configlib.add_arguments_dict(
    add_argument, components.inputlayers.image.configurable, prefix="image"
)
# ---
# Predinet Layer options
add_argument(
    "--relations",
    type=int,
    default=4,
    help="Number of relations to compute between features.",
)
add_argument(
    "--heads",
    type=int,
    default=4,
    help="Number of relation heads.",
)
add_argument(
    "--key_size",
    type=int,
    default=4,
    help="Number of relation heads.",
)
add_argument(
    "--output_hidden_size",
    type=int,
    default=8,
    help="MLP hidden layer size.",
)
# ---------------------------


class PrediNet(L.Layer):  # pylint: disable=too-many-instance-attributes
    """PrediNet model for supervised learning."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        relations: int = 4,
        heads: int = 4,
        key_size: int = 12,
        output_hidden_size: int = 8,
        num_classes: int = 2,
        **kwargs,
    ):
        """Initialise the PrediNet model.

        Args:
          relations: a scalar (int). Number of relations computed by each head.
          heads: a scalar (int). Number of PrediNet heads.
          key_size: a scalar (int). Size of the keys.
          output_hidden_size: a scalar (int). Size of hidden layer in output MLP.
          num_classes: a scalar (int). Number of classes in the output label.
        """
        super().__init__(**kwargs)
        # ---------------------------
        self.relations = relations
        self.heads = heads
        self.key_size = key_size
        self.output_hidden_size = output_hidden_size
        self.num_classes = num_classes

        # Define all model components
        self.get_keys = L.Dense(self.key_size, use_bias=False)
        self.get_query1 = L.Dense(self.heads * self.key_size, use_bias=False)
        self.get_query2 = L.Dense(self.heads * self.key_size, use_bias=False)
        self.embed_entities = L.Dense(self.relations, use_bias=False)
        self.output_hidden = L.Dense(self.output_hidden_size, activation="relu")
        self.output_layer = L.Dense(self.num_classes)

    def call(self, inputs, **kwargs):  # pylint: disable=too-many-locals
        """Applies model to inputs x yielding a label."""
        # inputs {'objects': (B, O, E), 'task_id': (B, T)}
        # task_id is optional
        batch_size = tf.shape(inputs["objects"])[0]  # B
        # ---------------------------
        # Keys
        keys = self.get_keys(inputs["objects"])
        # (B, O, K)
        keys = tf.repeat(keys[:, None], self.heads, axis=1)
        # (B, H, O, K)
        # ---
        # We need to flatten like this to avoid None dimensions and let
        # the query dense layers work with dynamically
        flat_objects = tf.reshape(
            inputs["objects"],
            [batch_size, tf.reduce_prod(inputs["objects"].shape[1:])],
        )
        # (B, O*E)
        # Queries
        query1 = self.get_query1(flat_objects)
        # (B, H*K)
        query1 = tf.reshape(query1, [batch_size, self.heads, self.key_size])
        # (B, H, K)
        query1 = tf.expand_dims(query1, 2)
        # (B, H, 1, K)

        query2 = self.get_query2(flat_objects)
        # (B, H*K)
        query2 = tf.reshape(query2, [batch_size, self.heads, self.key_size])
        # (B, H, K)
        query2 = tf.expand_dims(query2, 2)
        # (B, H, 1, K)
        # ---
        # Attention weights
        keys_t = tf.transpose(keys, perm=[0, 1, 3, 2])
        # (B, H, K, O)
        att1 = tf.nn.softmax(tf.matmul(query1, keys_t), -1)
        att2 = tf.nn.softmax(tf.matmul(query2, keys_t), -1)
        # (B, H, 1, O)
        # ---------------------------
        # Reshape features
        objects_repeated = tf.repeat(inputs["objects"][:, None], self.heads, axis=1)
        # (B, H, O, E)
        # Compute a pair of features using attention weights
        feature1 = tf.squeeze(tf.matmul(att1, objects_repeated), 2)
        feature2 = tf.squeeze(tf.matmul(att2, objects_repeated), 2)
        # (B, H, E)
        # ---
        # Spatial embedding
        embedding1 = self.embed_entities(feature1)
        embedding2 = self.embed_entities(feature2)
        # (B, H, R)
        # ---
        # Comparator
        relations = tf.subtract(embedding1, embedding2)
        # (B, H, R)
        # ---------------------------
        # Positions
        # We will not put positions back into the selected objects
        # because it is not a fair comparison to other models
        # pos1 = tf.slice(feature1, [0, 0, self._channels], [-1, -1, -1])
        # pos2 = tf.slice(feature2, [0, 0, self._channels], [-1, -1, -1])
        # (batch_size, heads, 2)
        # Collect relations and concatenate positions
        # relations = tf.concat([dx, pos1, pos2], 2)
        # (batch_size, heads, relations+4)
        # ---------------------------
        relations = tf.reshape(relations, [batch_size, self.heads * self.relations])
        # (B, H*R)
        # Append task id
        if "task_id" in inputs:
            relations = tf.concat([relations, inputs["task_id"]], -1)
            # (B, H*R+T)
        # ---
        # Apply output network
        hidden_activations = self.output_hidden(relations)
        output = self.output_layer(hidden_activations)
        # ---------------------------
        return {"label": output, "att1": att1, "att2": att2}

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update(
            {
                "relations": self.relations,
                "heads": self.heads,
                "key_size": self.key_size,
                "output_hidden_size": self.output_hidden_size,
                "num_classes": self.num_classes,
            }
        )
        return config


def process_image(image: tf.Tensor, _: Dict[str, Any]) -> tf.Tensor:
    """Process given image input extract objects."""
    # image (B, W, H, C)
    image_layer = utils.factory.get_and_init(
        components.inputlayers.image, C, "predinet_image_", name="image_layer"
    )
    raw_objects = image_layer(image)  # (B, W, H, E)
    return SpacialFlatten()(raw_objects)  # (B, O, E)


def process_task_id(task_id: tf.Tensor, input_desc: Dict[str, Any]) -> tf.Tensor:
    """Process given task ids."""
    return OneHotCategoricalInput(input_desc["num_categories"])(task_id)  # (B, T)


def build_model(task_description: Dict[str, Any]) -> Dict[str, Any]:
    """Build the predinet model."""
    # ---------------------------
    # Setup and process inputs
    processors = {"image": process_image, "task_id": process_task_id}
    predinet_inputs = utils.factory.create_input_layers(task_description, processors)
    # ---------------------------
    # Passed processed inputs to Predinet
    input_dict = {"objects": predinet_inputs["processed"]["image"]}
    if "task_id" in predinet_inputs["processed"]:
        input_dict["task_id"] = predinet_inputs["processed"]["task_id"]
    output_dict = PrediNet(
        relations=C["predinet_relations"],
        heads=C["predinet_heads"],
        key_size=C["predinet_key_size"],
        output_hidden_size=C["predinet_output_hidden_size"],
        num_classes=task_description["outputs"]["label"]["num_categories"],
    )(input_dict)
    output_dict = ReportLayer()(output_dict)
    # ---------------------------
    # Create model instance
    model = tf.keras.Model(
        inputs=predinet_inputs["input_layers"],
        outputs={"label": output_dict["label"]},
        name="predinet_model",
    )
    # ---------------------------
    # Compile model for training
    dataset_type = task_description["outputs"]["label"]["type"]
    assert (
        dataset_type == "binary"
    ), f"PrediNet setup requires a binary classification dataset, got {dataset_type}"
    loss = {"label": tf.keras.losses.BinaryCrossentropy(from_logits=True)}
    metrics = {"label": tf.keras.metrics.BinaryAccuracy(name="acc")}
    # ---------------------------
    return {"model": model, "loss": loss, "metrics": metrics}
