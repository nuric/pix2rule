"""Rule learning model for relsgame dataset."""
import tensorflow as tf
import tensorflow.keras.layers as L
import numpy as np

from configlib import config as C
from reportlib import report
from components.relsgame_cnn import RelationsGameCNN

# from components.shuffle import Shuffle
from components import ops
from .rule_learner import RelsgameRuleLearner


class ObjectSelection(L.Layer):
    """Select a subset of objects based on object score."""

    def __init__(self, num_select: int = 4, **kwargs):
        super(ObjectSelection, self).__init__(**kwargs)
        self.num_select = num_select
        self.object_score = L.Dense(1, activation="sigmoid")

    def call(self, inputs, **kwargs):
        """Perform forward pass."""
        # inputs (B, num_objects O, embedding_size E)
        # ---------------------------
        object_scores = self.object_score(inputs)  # (B, O, 1)
        object_scores = tf.squeeze(object_scores, -1)  # (B, O)
        report["object_scores"] = object_scores
        # ---------------------------
        # Do a left to right selection
        atts = list()
        last_select = object_scores
        for _ in range(self.num_select):
            lr_reduction = ops.leftright_cumprod(last_select)  # (B, O)
            atts.append(lr_reduction)
            # We can't select object again
            last_select *= 1 - lr_reduction
        object_atts = tf.stack(atts, 1)  # (B, N, O)
        report["object_atts"] = object_atts
        # ---------------------------
        # Select the objects based on the attention
        # (B, N, O) x (B, O, E) -> (B, N, E)
        selected_objects = tf.einsum("bno,boe->bne", object_atts, inputs)
        return selected_objects

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super(ObjectSelection, self).get_config()
        config.update({"num_select": self.num_select})
        return config


class RelsgameFeatures(L.Layer):
    """Relations game object features."""

    def __init__(
        self,
        num_unary: int = 4,
        num_unary_disjuncts: int = 4,
        num_binary: int = 4,
        num_binary_disjuncts: int = 4,
        **kwargs
    ):
        super(RelsgameFeatures, self).__init__(**kwargs)
        self.num_unary_with_disjuncts = (num_unary, num_unary_disjuncts)
        self.num_binary_with_disjucts = (num_binary, num_binary_disjuncts)
        self.unary_model = L.Dense(num_unary * num_unary_disjuncts, name="unary_model")
        self.binary_model = L.Dense(
            num_binary * num_binary_disjuncts, name="binary_model"
        )
        # There are 4 tasks, we'll one hot encode them
        self.num_tasks = 5
        self.task_embedding = L.Embedding(
            self.num_tasks,
            self.num_tasks,
            mask_zero=False,
            weights=[tf.eye(self.num_tasks)],
            trainable=False,
        )

    def call(self, inputs, **kwargs):
        """Perform forward pass."""
        # inputs {'objects': (B, num_objects O, embedding_size E), task_id: (B,)}
        # ---------------------------
        # Compute unary features
        unary_feats = self.unary_model(inputs["objects"])  # (B, O, P1)
        ushape = tf.shape(unary_feats)  # (B, O, P1)
        unary_feats = tf.reshape(
            unary_feats, tf.concat([ushape[:2], self.num_unary_with_disjuncts], 0)
        )  # (B, O, U, V)
        unary_feats = tf.nn.softmax(unary_feats, -1)  # (B, O, U, V)
        unary_feats = tf.reshape(unary_feats, ushape)  # (B, O, P1)
        unary_feats = tf.pad(
            unary_feats, [[0, 0], [0, 0], [0, self.num_tasks]], constant_values=0.0
        )  # (B, O, P1+T)
        # ---
        task_embed = self.task_embedding(inputs["task_id"])  # (B, T)
        task_embed = tf.pad(
            task_embed,
            [[0, 0], [np.prod(self.num_unary_with_disjuncts), 0]],
            constant_values=0.0,
        )  # (B, P1+T)
        # ---
        unary_feats = tf.concat([unary_feats, task_embed[:, None]], 1)  # (B, O+1, P1+T)
        # ---------------------------
        # Compute binary features
        arg1 = inputs["objects"][:, :, None]  # (B, O, 1, E)
        arg2 = inputs["objects"][:, None]  # (B, 1, O, E)
        paired_objects = tf.concat(
            [arg1 + arg2, arg1 - arg2, arg1 * arg2], -1
        )  # (B, O, O, 3E)
        binary_feats = self.binary_model(paired_objects)  # (B, O, O, P2)
        bshape = tf.shape(binary_feats)  # (B, O, O, P2)
        binary_feats = tf.reshape(
            binary_feats, tf.concat([bshape[:3], self.num_binary_with_disjucts], 0)
        )  # (B, O, O, U, V)
        binary_feats = tf.nn.softmax(binary_feats, -1)  # (B, O, O, U, V)
        binary_feats = tf.reshape(binary_feats, bshape)  # (B, O, O, P2)
        # ---
        # Append empty task node which has no binary relations with other objects
        binary_feats = tf.pad(
            binary_feats, [[0, 0], [0, 1], [0, 1], [0, 0]], constant_values=0.0
        )  # (B, O+1, O+1, P2)
        # ---------------------------
        return unary_feats, binary_feats

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super(RelsgameFeatures, self).get_config()
        config.update(
            {
                "num_unary_with_disjuncts": self.num_unary_with_disjuncts,
                "num_binary_with_disjucts": self.num_binary_with_disjucts,
            }
        )
        return config


def build_model() -> tf.keras.Model:  # pylint: disable=too-many-locals
    """Build the trainable model."""
    # ---------------------------
    # Setup inputs
    image = L.Input(shape=(36, 36, 3), name="image", dtype="float32")  # (B, W, H, C)
    task_id = L.Input(shape=(), name="task_id", dtype="int32")  # (B,)
    inv_image = L.Input(
        shape=(36, 36, 3), name="inv_image", dtype="float32"
    )  # (I, W, H, C)
    inv_task_id = L.Input(shape=(), name="inv_task_id", dtype="int32")  # (I,)
    inv_label = L.Input(shape=(), name="inv_label", dtype="int32")  # (I,)
    # ---------------------------
    # Process the images
    visual_layer = RelationsGameCNN()
    # shuffle_layer = Shuffle()
    raw_objects = visual_layer(image)  # (B, O, E)
    inv_raw_objects = visual_layer(inv_image)  # (I, O, E)
    # raw_objects = shuffle_layer(raw_objects)  # (B, O, E)
    # inv_raw_objects = shuffle_layer(inv_raw_objects)  # (I, O, E)
    # ---------------------------
    # Select a subset of objects
    obj_selector = ObjectSelection()
    selected_objects = obj_selector(raw_objects)  # (B, N, O)
    inv_selected_objects = obj_selector(inv_raw_objects)  # (I, N, E)
    # ---------------------------
    # Extract unary and binary features
    feat_ext = RelsgameFeatures()
    unary_feats, binary_feats = feat_ext(
        {"objects": selected_objects, "task_id": task_id}
    )  # (B, N, P1), (B, N, N, P2)
    inv_unary_feats, inv_binary_feats = feat_ext(
        {"objects": inv_selected_objects, "task_id": inv_task_id}
    )  # (I, N, P1), (I; N, N, P2)
    # ---------------------------
    # Perform rule learning and get predictions
    combined = {
        "unary_feats": unary_feats,
        "binary_feats": binary_feats,
        "inv_unary_feats": inv_unary_feats,
        "inv_binary_feats": inv_binary_feats,
        "inv_label": inv_label,
    }
    predictions = RelsgameRuleLearner(max_invariants=C["max_invariants"])(
        combined
    )  # (B, S)
    return tf.keras.Model(
        inputs=[image, task_id, inv_image, inv_task_id, inv_label],
        outputs=predictions,
        name="relsgame_model",
    )
