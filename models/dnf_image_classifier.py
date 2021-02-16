"""Rule learning model for relsgame dataset."""
from typing import Dict, Any
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L

from reportlib import ReportLayer
from components.relsgame_cnn import RelationsGameCNN
from components.object_selection import RelaxedObjectSelection
from .dnf_layer import DNFLayer


class RelsgameFeatures(L.Layer):
    """Relations game object features."""

    def __init__(self, unary_preds: int = 4, binary_preds: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.unary_preds = unary_preds
        self.binary_preds = binary_preds
        self.unary_model = L.Dense(
            unary_preds,
            activation="sigmoid",
            name="unary_model",
            bias_initializer=tf.keras.initializers.Constant(4),
            # bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=4.0),
        )
        self.binary_model = L.Dense(
            binary_preds,
            activation="sigmoid",
            name="binary_model",
            bias_initializer=tf.keras.initializers.Constant(4),
            # bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=4.0),
        )
        # There are 5 tasks, we'll one hot encode them
        self.num_tasks = 5

    def build(self, input_shape: Dict[str, tf.TensorShape]):
        """Build layer weights."""
        # input_shape {'objects': (B, num_objects N, embedding_size E), task_id: (B,)}
        num_objects = input_shape["objects"][1]  # N
        # The following captures all the non-diagonal indices representing p(X,Y)
        # and omitting p(X,X). So every object compared to every other object
        binary_idxs = np.stack(np.nonzero(1 - np.eye(num_objects))).T  # (O*(O-1), 2)
        # pylint: disable=attribute-defined-outside-init
        self.binary_idxs = np.reshape(
            binary_idxs, (num_objects, num_objects - 1, 2)
        )  # (O, O-1, 2)

    def call(self, inputs: Dict[str, tf.Tensor], **kwargs):
        """Perform forward pass."""
        # inputs {'objects': (B, num_objects N, embedding_size E), task_id: (B,)}
        # ---------------------------
        # Compute nullary predicates
        task_embed = tf.eye(self.num_tasks)  # (task, task)
        nullary_preds = tf.gather(task_embed, inputs["task_id"], axis=0)  # (B, P0)
        # ---------------------------
        # Compute unary features
        unary_preds = self.unary_model(inputs["objects"])  # (B, O, P1)
        # ---------------------------
        # Compute binary features
        arg1 = tf.gather(
            inputs["objects"], self.binary_idxs[..., 0], axis=1
        )  # (B, O, O-1, E)
        arg2 = tf.gather(
            inputs["objects"], self.binary_idxs[..., 1], axis=1
        )  # (B, O, O-1, E)
        paired_objects = arg1 - arg2  # (B, O, O-1, E)
        binary_preds = self.binary_model(paired_objects)  # (B, O, O-1, P2)
        # ---------------------------
        return {
            "nullary_preds": nullary_preds,
            "unary_preds": unary_preds,
            "binary_preds": binary_preds,
        }

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update(
            {
                "unary_preds": self.unary_preds,
                "binary_preds": self.binary_preds,
            }
        )
        return config


def build_model(
    data_description: Dict[str, Any]
) -> tf.keras.Model:  # pylint: disable=too-many-locals
    """Build the trainable model."""
    # ---------------------------
    # Setup inputs
    image = L.Input(shape=(12, 12, 3), name="image", dtype="float32")  # (B, W, H, C)
    task_id = L.Input(shape=(), name="task_id", dtype="int32")  # (B,)
    # ---------------------------
    # Process the images
    raw_objects = RelationsGameCNN()(image)  # (B, O, E)
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
    feat_ext = RelsgameFeatures()
    ground_facts = feat_ext(
        {"objects": selected_objects["objects"], "task_id": task_id}
    )  # {'nullary_preds': (B, P0), 'unary_preds': (B, N, P1), 'binary_preds': (B, N, N-1, P2)}
    # ---------------------------
    # Perform rule learning and get predictions
    # ground_facts = AndLayer(arities=list(np.repeat([0, 1, 2], 3)), residiual=True)(
    #     ground_facts
    # )  # {'nullary_preds': ..., 'unary_preds': ..., 'binary_preds': ...}
    # ground_facts = AndLayer(arities=list(np.repeat([0, 1, 2], 3)), residiual=True)(
    #     ground_facts
    # )  # {'nullary_preds': ..., 'unary_preds': ..., 'binary_preds': ...}
    dnf_layer = DNFLayer(arities=[0, 0, 0, 0], recursive=True)
    padded_facts = dnf_layer.pad_inputs(
        ground_facts
    )  # {'nullary_preds': (B, P0 + R0), ...}
    for _ in range(2):
        padded_facts = dnf_layer(padded_facts)  # {'nullary_preds': (B, P0+R0), ...}
    predictions = padded_facts["nullary_preds"][..., -4:]  # (B, R0)
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
