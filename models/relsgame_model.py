"""Rule learning model for relsgame dataset."""
from typing import Dict
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_probability as tfp

from reportlib import report_tensor
from components.relsgame_cnn import RelationsGameCNN
from .rule_learner import AndLayer


class ObjectSelection(L.Layer):
    """Select a subset of objects based on object score."""

    def __init__(self, num_select: int = 2, initial_temperature: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_select = num_select
        self.initial_temperature = initial_temperature
        # We use a higher starting bias value so that the score inversion is more stable
        self.object_score = L.Dense(
            1, bias_initializer=tf.keras.initializers.Constant(10)
        )
        self.temperature = self.add_weight(
            name="temperature",
            initializer=tf.keras.initializers.Constant(initial_temperature),
            trainable=False,
        )

    def call(self, inputs, **kwargs):
        """Perform forward pass."""
        # inputs (B, num_objects O, embedding_size E)
        # ---------------------------
        object_scores = self.object_score(inputs)  # (B, O, 1)
        object_scores = tf.squeeze(object_scores, -1)  # (B, O)
        report_tensor("object_scores", object_scores)
        # ---------------------------
        # TopK selection
        # _, idxs = tf.math.top_k(object_scores, k=self.num_select)  # (B, N)
        # return tf.gather(inputs, idxs, axis=1, batch_dims=1)  # (B, N, O)
        # ---------------------------
        # Do a left to right selection
        atts = list()
        last_select = object_scores
        for _ in range(self.num_select):
            sample = tfp.distributions.RelaxedOneHotCategorical(
                self.temperature, logits=last_select
            ).sample()  # (B, O)
            sample = tf.cast(sample, tf.float32)  # (B, O)
            atts.append(sample)
            last_select = sample * (-last_select) + (1 - sample) * last_select
        object_atts = tf.stack(atts, 1)  # (B, N, O)
        report_tensor("object_atts", object_atts)
        # ---------------------------
        # Select the objects based on the attention
        # (B, N, O) x (B, O, E) -> (B, N, E)
        selected_objects = tf.einsum("bno,boe->bne", object_atts, inputs)
        return selected_objects

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update(
            {
                "num_select": self.num_select,
                "initial_temperature": self.initial_temperature,
            }
        )
        return config


class RelsgameFeatures(L.Layer):
    """Relations game object features."""

    def __init__(self, unary_preds: int = 4, binary_preds: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.unary_preds = unary_preds
        self.binary_preds = binary_preds
        self.unary_model = L.Dense(unary_preds, activation="tanh", name="unary_model")
        self.binary_model = L.Dense(
            binary_preds, activation="tanh", name="binary_model"
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
        task_embed = tf.eye(self.num_tasks) * 2 - 1  # (task, task)
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
        paired_objects = tf.concat(
            [arg1 + arg2, arg1 - arg2, arg1 * arg2], -1
        )  # (B, O, O-1, 3E)
        binary_preds = self.binary_model(paired_objects)  # (B, O, O-1, E)
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


def build_model() -> tf.keras.Model:  # pylint: disable=too-many-locals
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
    obj_selector = ObjectSelection()
    # obj_selector = SlotAttention(
    # num_iterations=3, num_slots=3, slot_size=32, mlp_hidden_size=32
    # )
    selected_objects = obj_selector(raw_objects)  # (B, N, E)
    # ---------------------------
    # Extract unary and binary features
    feat_ext = RelsgameFeatures()
    ground_facts = feat_ext(
        {"objects": selected_objects, "task_id": task_id}
    )  # {'nullary_preds': (B, P0), 'unary_preds': (B, N, P1), 'binary_preds': (B, N, N-1, P2)}
    # ---------------------------
    # Perform rule learning and get predictions
    predictions = AndLayer()(ground_facts)  # (B, S)
    return tf.keras.Model(
        inputs=[image, task_id],
        outputs=predictions,
        name="relsgame_model",
    )
