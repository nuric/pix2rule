"""Unification MLP."""
import os
import logging
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L

import configlib
from configlib import config as C
import datasets

import unification
from utils.ops import reduce_probsum, leftright_cumprod
import utils.ops as ops

# Calm down tensorflow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Setup logging
logging.getLogger().setLevel(logging.INFO)

# Disable scientific printing
np.set_printoptions(suppress=True, precision=5, linewidth=180)

# ---------------------------

# Arguments
parser = configlib.add_parser("UMLP options.")
parser.add_argument(
    "--invariants", default=1, type=int, help="Number of invariants per task."
)
parser.add_argument("--embed", default=16, type=int, help="Embedding size.")
parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
parser.add_argument("--nouni", action="store_true", help="Disable unification.")

# Store in global config object inside configlib
configlib.parse()
print("Running with configuration:")
configlib.print_config()

# Tensorflow graph mode (i.e. tf.function)
tf.config.experimental_run_functions_eagerly(C["debug"])

# ---------------------------


class GraphFeatures(L.Layer):
    """Compute graph features of a given input."""

    def __init__(self, **kwargs):
        super(GraphFeatures, self).__init__(**kwargs)
        # self.embedding = L.Embedding(C["seq_symbols"], C["embed"], mask_zero=True)
        self.embedding = L.Embedding(
            1 + C["seq_symbols"],
            1 + C["seq_symbols"],
            mask_zero=True,
            weights=[tf.eye(1 + C["seq_symbols"])],
            trainable=False,
        )

    def build(self, input_shape):
        """Build the layer parameters."""
        pass

    def call(self, inputs, **kwargs):
        """Perform forward pass."""
        # ---------------------------
        # Compute unary predicates
        neye = tf.eye(1 + C["seq_length"])  # (N, N)
        unary_p_pos = tf.repeat(neye[None], inputs.shape[0], axis=0)  # (B, N, N)
        unary_p_sym = self.embedding(inputs)  # (B, N, S)
        unary_ps = tf.concat([unary_p_pos, unary_p_sym], axis=-1)  # (B, N, P1)
        # ---------------------------
        # Compute binary predicates
        binary_eq_sym = tf.matmul(
            unary_p_sym, unary_p_sym, transpose_b=True
        )  # (B, N, N)
        # Collect binary predicates
        binary_ps = tf.expand_dims(binary_eq_sym, -1)  # (B, N, N, P2)
        # Remove self relations, i.e. p(X, X) = 0 always
        # p(X,X) should be modelled as p(X) (unary) instead
        binary_ps *= 1 - neye[:, :, None]  # (B, N, N, P2)
        # ---------------------------
        return unary_ps, binary_ps

    def get_config(self):
        """Return serialisation configuration."""
        config = super().get_config()
        config.update({"num_invariants": self.num_invariants})
        return config


class UMLP(tf.keras.Model):  # pylint: disable=too-many-ancestors
    """Unification MLP Model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph_feats = GraphFeatures()
        invs = [
            [1, 2, 4, 3, 4],
            [2, 3, 4, 5, 1],
            [3, 2, 4, 4, 7],
            [4, 1, 1, 2, 3],
            [4, 2, 1, 2, 3],
            [4, 3, 1, 2, 3],
            [4, 5, 4, 4, 6],
            [4, 5, 6, 2, 6],
            [4, 2, 5, 6, 6],
        ]
        labels = [2, 3, 7, 1, 2, 3, 4, 6, 6]
        # invs = [[1, 2, 4, 3, 4]]
        # labels = [2]
        self.inv_inputs = tf.constant(invs, dtype=tf.int32)  # (I, 1L)
        self.inv_labels = tf.constant(labels, dtype=tf.int32)  # (I,)
        seq_len = 1 + C["seq_length"]  # 1+ for task id
        self.inv_unaryp = self.add_weight(
            name="inv_unaryp",
            shape=(len(invs), seq_len, seq_len + 1 + C["seq_symbols"]),
            initializer=tf.keras.initializers.RandomNormal(mean=0),
            trainable=True,
        )  # (I, 1L, 1L, P1)
        self.inv_binaryp = self.add_weight(
            name="inv_binaryp",
            shape=(len(invs), seq_len, seq_len, 1),
            initializer=tf.keras.initializers.RandomNormal(mean=0),
            trainable=True,
        )  # (I, 1L, 1L, P2)
        self.inv_out_map = self.add_weight(
            name="inv_out_map",
            shape=(len(invs), seq_len + 1),
            initializer="random_normal",
            trainable=True,
        )  # (I, IL+1)
        self.syms_eye = tf.eye(1 + C["seq_symbols"])  # (S, S)

    def call(self, inputs: tf.Tensor, training=None, mask=None):
        """Perform forward pass of the model."""
        # inputs (B, 1L)
        report = dict()
        # ---------------------------
        # Compute invariant features
        inv_unary_feats, inv_binary_feats = self.graph_feats(
            self.inv_inputs
        )  # (I, IL, P1), # (I, IL, IL, P2)
        # Gather unary invariant map
        inv_unary_map = tf.nn.sigmoid(self.inv_unaryp, -1)  # (I, IL, P1)
        # Gather binary invariant map
        inv_binary_map = tf.nn.sigmoid(self.inv_binaryp, -1)  # (I, IL, IL, P2)
        # The examples need to satisfy these conditions
        inv_unary_conds = inv_unary_map * inv_unary_feats  # (I, IL, P1)
        inv_binary_conds = inv_binary_map * inv_binary_feats  # (I, IL, IL, P2)
        report["inv_unary_conds"] = inv_unary_conds
        report["inv_binary_conds"] = inv_binary_conds
        inv_loss = tf.reduce_sum(inv_unary_conds, [1, 2]) + tf.reduce_sum(
            inv_binary_conds, [1, 2, 3]
        )  # (I,)
        inv_loss = tf.reduce_mean(inv_loss) * 0.01  # ()
        self.add_loss(inv_loss)
        # ---------------------------
        # Compute batch features
        batch_unary_feats, batch_binary_feats = self.graph_feats(
            inputs
        )  # (B, BL, P1), (B, BL, BL, P2)
        # ---------------------------
        # Unify
        uni_sets = unification.unify(
            inv_unary_conds,
            inv_binary_conds,
            batch_unary_feats,
            batch_binary_feats,
            iterations=1,
        )  # (B, I, IL, BL)
        report["uni_sets"] = uni_sets
        # ---------------------------
        # Compute output edges
        # null_pred = self.syms_eye[0]  # (S,)
        # node_uni = uni_sets * 0.99
        # node_select = tf.nn.softmax(
        # tf.math.log(node_uni / (1 - node_uni)), -1
        # )  # (B, I, IL, BL)
        node_select = tf.nn.softmax(uni_sets * 10, -1)  # (B, I, IL, BL)
        # (B, I, IL, BL) x (B, BL, S) -> (B, I, IL, S)
        edge_outs = tf.einsum("bilk,bku->bilu", node_select, batch_unary_feats[..., 5:])
        # ---------------------------
        # Select the output node
        # Either a node in the output, or the constant output node
        inv_out_map = tf.nn.softmax(self.inv_out_map, -1)  # (I, IL+1)
        report["inv_out_map"] = inv_out_map
        # (I, IL) x (B, I, IL, S) -> (B, I, S)
        inv_nodes_out = tf.einsum("il,bilp->bip", inv_out_map[:, :-1], edge_outs)
        inv_const_out = tf.gather(self.syms_eye, self.inv_labels)  # (I, S)
        inv_sym_out = inv_nodes_out + inv_out_map[:, -1:] * inv_const_out  # (B, I, S)
        report["inv_sym_out"] = inv_sym_out
        # ---------------------------
        # Find which invariants unify
        inv_uni = tf.reduce_prod(reduce_probsum(uni_sets, -1), -1)  # (B, I)
        # inv_uni = tf.concat([inv_uni, tf.ones((inv_uni.shape[0], 1))], -1)  # (B, I+1)
        # inv_select = leftright_cumprod(inv_uni)  # (B, I+1)
        # inv_select = tf.nn.softmax(tf.math.log(inv_uni / (1 - inv_uni)))  # (B, I)
        inv_select = tf.nn.softmax(inv_uni * 10)  # (B, I)
        report["inv_select"] = inv_select
        # ---------------------------
        # (B, I) x (B, I, S) -> (B, S)
        predictions = tf.einsum("bi,bis->bs", inv_select, inv_sym_out)
        # inv_preds = tf.einsum("bi,bis->bs", inv_select[:, :-1], inv_sym_out)
        # predictions = inv_preds + inv_select[:, -1:] * null_pred  # (B, S)
        report["predictions"] = predictions
        # ---------------------------
        return report


# ---------------------------


def train_step(model, batch, lossf, optimiser):
    """Perform one batch update."""
    # batch = {'input': (B, 1+L), 'label': (B,)}
    report = dict()
    with tf.GradientTape() as tape:
        report = model(batch["input"], training=True)  # {'predictions': (B, S), ...}
        # labels = tf.repeat(batch["label"][:, None], 9, 1)  # (B, I)
        loss = lossf(batch["label"], report["predictions"])  # (B, I)
        loss += sum(model.losses)  # Keras accumulated losses, e.g. regularisers
        # loss = lossf(labels, report["predictions"])  # (B, I)
        # loss *= report["inv_select"]
        # loss += (1 - reduce_probsum(report["inv_select"], -1, keepdims=True)) * 2.0
        # loss = tf.reduce_mean(loss)
        report["loss"] = loss
    gradients = tape.gradient(loss, model.trainable_variables)
    # if any([tf.reduce_any(tf.math.is_nan(g)) for g in gradients if g is not None]):
    # import ipdb

    # ipdb.set_trace()
    # print("HERE")
    optimiser.apply_gradients(zip(gradients, model.trainable_variables))
    return report


def train():
    """Training loop."""
    # Load data
    dsets = datasets.sequences.load_data()
    print(dsets)
    # ---------------------------
    # Setup model
    model = UMLP()
    # ---------------------------
    # Setup metrics
    metrics = {
        k: {
            "loss": tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=False),
            "acc": tf.keras.metrics.SparseCategoricalAccuracy(),
        }
        for k in dsets.keys()
    }
    # ---------------------------
    # Training loop
    lossf = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimiser = tf.keras.optimizers.Adam()
    for i, batch in dsets["train"].enumerate():
        # batch = {'input': (B, 1+L), 'label': (B,)}
        # batch = {"input": tf.constant([[1, 2, 4, 3, 4]]), "label": tf.constant([2])}
        report = train_step(model, batch, lossf, optimiser)
        if tf.math.is_nan(report["loss"]):
            print("Loss is NaN.")
            break
        if i.numpy() % 100 == 0:
            print(i.numpy(), report["loss"].numpy())
        if i.numpy() == 20000 or report["loss"].numpy() < 0.001:
            print("Converged or complete.")
            break
    import ipdb

    ipdb.set_trace()
    print("HERE")


if __name__ == "__main__":
    train()
