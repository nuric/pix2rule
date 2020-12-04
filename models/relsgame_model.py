"""Rule learning model for relsgame dataset."""
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_probability as tfp

from reportlib import report_tensor
from components.relsgame_cnn import RelationsGameCNN
from .rule_learner import BaseRuleLearner


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

    def __init__(self, latent_size: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.latent_size = latent_size
        self.unary_model = L.Dense(latent_size, name="unary_model")
        self.binary_model = L.Dense(latent_size, name="binary_model")
        # There are 4 tasks, we'll one hot encode them
        self.num_tasks = 5
        self.task_embedding = L.Embedding(
            self.num_tasks,
            latent_size,
            mask_zero=False,
            trainable=True,
        )

    def call(self, inputs, **kwargs):
        """Perform forward pass."""
        # inputs {'objects': (B, num_objects O, embedding_size E), task_id: (B,)}
        # ---------------------------
        # Compute unary features
        unary_feats = self.unary_model(inputs["objects"])  # (B, O, E)
        task_embed = self.task_embedding(inputs["task_id"])  # (B, E)
        unary_feats = tf.concat([unary_feats, task_embed[:, None]], 1)  # (B, O+1, E)
        # ---------------------------
        # Compute binary features
        arg1 = inputs["objects"][:, :, None]  # (B, O, 1, E)
        arg2 = inputs["objects"][:, None]  # (B, 1, O, E)
        paired_objects = tf.concat(
            [arg1 + arg2, arg1 - arg2, arg1 * arg2], -1
        )  # (B, O, O, 3E)
        binary_feats = self.binary_model(paired_objects)  # (B, O, O, E)
        # Mask out self relations p(X,X) = 0
        binary_feats *= 1 - tf.eye(tf.shape(binary_feats)[1])[..., None]  # (B, O, O, E)
        # ---
        # Append empty task node which has no binary relations with other objects
        binary_feats = tf.pad(
            binary_feats, [[0, 0], [0, 1], [0, 1], [0, 0]], constant_values=0.0
        )  # (B, O+1, O+1, E)
        # ---------------------------
        return unary_feats, binary_feats

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update(
            {
                "latent_size": self.latent_size,
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
    unary_feats, binary_feats = feat_ext(
        {"objects": selected_objects, "task_id": task_id}
    )  # (B, N, P1), (B, N, N, P2)
    # ---------------------------
    # Perform rule learning and get predictions
    combined = {
        "unary_feats": unary_feats,
        "binary_feats": binary_feats,
    }
    predictions = BaseRuleLearner()(combined)  # (B, S)
    return tf.keras.Model(
        inputs=[image, task_id],
        outputs=predictions,
        name="relsgame_model",
    )
