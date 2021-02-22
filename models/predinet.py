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

# Setup configurable parameters of the model
add_argument = configlib.add_group("Predinet Image Model Options.", prefix="predinet")
# ---
# Image layer parameters
add_argument(
    "--add_image_noise_stddev",
    type=float,
    default=0.0,
    help="Optional noise to add image input before processing.",
)
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


class PrediNet(L.Layer):
    """PrediNet model for supervised learning."""

    def __init__(
        self,
        relations=4,
        heads=4,
        key_size=12,
        output_hidden_size=8,
        num_classes=2,
        **kwargs,
    ):
        """Initialise the PrediNet model.

        Args:
          resolution: a scalar (int). Resolution of raw images.
          conv_out_size: a scalar (int). Downsampled image resolution obtained at
            the output of the convolutional layer.
          filter_size: a scalar (int). Filter size for the convnet.
          stride: a scalar (int). Stride size for the convnet.
          channels: a scalar (int). Number of channels of the convnet.
          relations: a scalar (int). Number of relations computed by each head.
          heads: a scalar (int). Number of PrediNet heads.
          key_size: a scalar (int). Size of the keys.
          output_hidden_size: a scalar (int). Size of hidden layer in output MLP.
          num_classes: a scalar (int). Number of classes in the output label.
          num_tasks: a scalar (int). Max number of possible tasks.
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

    def call(self, inputs, **kwargs):
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
        return {"output": output, "att1": att1, "att2": att2}


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


def build_model(  # pylint: disable=too-many-locals
    task_description: Dict[str, Any]
) -> Dict[str, Any]:
    """Build the predinet model."""
    # ---------------------------
    # Setup and process inputs
    input_layers = dict()
    processed = dict()
    for input_name, input_desc in task_description["inputs"].items():
        input_layer = L.Input(
            shape=input_desc["shape"][1:],
            name=input_name,
            dtype=input_desc["dtype"].name,
        )
        input_layers[input_name] = input_layer
        if input_desc["type"] == "image":
            image_layer = get_and_init(
                components.inputlayers.image, "predinet_image_", name="image_layer"
            )
            raw_objects = image_layer(input_layer)  # (B, W, H, E)
            raw_objects = SpacialFlatten()(raw_objects)  # (B, O, E)
            processed["objects"] = raw_objects
        elif input_desc["type"] == "categorical":
            processed["task_id"] = OneHotCategoricalInput(input_desc["num_categories"])(
                input_layer
            )
    # ---------------------------
    # Passed processed inputs to Predinet
    output_dict = PrediNet(
        relations=C["predinet_relations"],
        heads=C["predinet_heads"],
        key_size=C["predinet_key_size"],
        output_hidden_size=C["predinet_output_hidden_size"],
        num_classes=task_description["output"]["num_categories"],
    )(processed)
    output_dict = ReportLayer()(output_dict)
    # ---------------------------
    # Create model instance
    model = tf.keras.Model(
        inputs=input_layers,
        outputs=output_dict["output"],
        name="predinet_model",
    )
    # ---------------------------
    # Compile model for training
    dataset_type = task_description["output"]["type"]
    assert (
        dataset_type == "multiclass"
    ), f"DNF classifier requires a multiclass dataset, got {dataset_type}"
    loss = (tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
    # ---------------------------
    return {"model": model, "loss": loss, "metrics": metrics}
