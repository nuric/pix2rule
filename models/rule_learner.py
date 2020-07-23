"""Differentiable rule learning component."""
import tensorflow as tf

from configlib import config as C
from reportlib import report
import unification
from utils import ops
from .seq_feats import SequenceFeatures


class RuleLearner(
    tf.keras.layers.Layer
):  # pylint: disable=too-many-instance-attributes
    """Differentiable rule learning component."""

    def __init__(self, inv_inputs, inv_labels, num_symbols: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.graph_feats = SequenceFeatures()
        self.inv_inputs = tf.constant(inv_inputs, dtype=tf.int32)  # (I, 1L)
        self.inv_labels = tf.constant(inv_labels, dtype=tf.int32)  # (I,)
        self.num_symbols = num_symbols  # S
        self.syms_eye = tf.eye(1 + num_symbols)  # (S, S)

    def build(self, input_shape: tf.TensorShape):
        """Build layer weights."""
        # pylint: disable=attribute-defined-outside-init
        seq_len = input_shape[-1]  # 1+ for task id
        ilen = self.inv_inputs.shape[0]  # I
        self.inv_unaryp = self.add_weight(
            name="inv_unaryp",
            shape=(ilen, seq_len, seq_len + 1 + self.num_symbols),
            initializer=tf.keras.initializers.RandomNormal(mean=0),
            trainable=True,
        )  # (I, 1L, 1L, P1)
        self.inv_binaryp = self.add_weight(
            name="inv_binaryp",
            shape=(ilen, seq_len, seq_len, 1),
            initializer=tf.keras.initializers.RandomNormal(mean=0),
            trainable=True,
        )  # (I, 1L, 1L, P2)
        self.inv_out_map = self.add_weight(
            name="inv_out_map",
            shape=(ilen, seq_len + 1),
            initializer="random_normal",
            trainable=True,
        )  # (I, IL+1)

    def call(self, inputs: tf.Tensor, **kwargs):
        """Perform forward pass of the model."""
        # pylint: disable=too-many-locals
        # inputs (B, 1L)
        # ---------------------------
        # Compute invariant features
        inv_unary_feats, inv_binary_feats = self.graph_feats(
            self.inv_inputs
        )  # (I, IL, P1), # (I, IL, IL, P2)
        # Gather unary invariant map
        inv_unary_map = tf.nn.sigmoid(self.inv_unaryp)  # (I, IL, P1)
        # Gather binary invariant map
        inv_binary_map = tf.nn.sigmoid(self.inv_binaryp)  # (I, IL, IL, P2)
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
        alpha = 0.999  # softmax target scale on one-hot vector
        sm_scale = tf.math.log(
            alpha
            * tf.cast((tf.shape(self.inv_inputs)[0] - 1), tf.float32)
            / (1 - alpha)
        )  # k = log(alpha * (n-1) / (1-alpha)) derived from softmax(kx)
        node_select = tf.nn.softmax(uni_sets * sm_scale, -1)  # (B, I, IL, BL)
        # (B, I, IL, BL) x (B, BL, S) -> (B, I, IL, S)
        edge_outs = tf.einsum("bilk,bku->bilu", node_select, batch_unary_feats[..., 5:])
        # ---------------------------
        # Select the output node
        # Either a node in the output, or the constant output node
        inv_out_map = tf.nn.softmax(self.inv_out_map, -1)  # (I, IL+1)
        report["inv_out_map"] = inv_out_map
        # (I, IL) x (B, I, IL, S) -> (B, I, S)
        inv_nodes_out = tf.einsum("il,bilp->bip", inv_out_map[:, :-1], edge_outs)
        inv_const_out = tf.gather(self.syms_eye, self.inv_labels, axis=0)  # (I, S)
        inv_sym_out = inv_nodes_out + inv_out_map[:, -1:] * inv_const_out  # (B, I, S)
        report["inv_sym_out"] = inv_sym_out
        # ---------------------------
        # Find which invariants unify
        inv_uni = tf.reduce_prod(ops.reduce_probsum(uni_sets, -1), -1)  # (B, I)
        # inv_uni = tf.concat([inv_uni, tf.ones((inv_uni.shape[0], 1))], -1)  # (B, I+1)
        # inv_select = leftright_cumprod(inv_uni)  # (B, I+1)
        # inv_select = tf.nn.softmax(tf.math.log(inv_uni / (1 - inv_uni)))  # (B, I)
        inv_select = tf.nn.softmax(inv_uni * sm_scale)  # (B, I)
        report["inv_select"] = inv_select
        # ---------------------------
        # (B, I) x (B, I, S) -> (B, S)
        predictions = tf.einsum("bi,bis->bs", inv_select, inv_sym_out)
        # inv_preds = tf.einsum("bi,bis->bs", inv_select[:, :-1], inv_sym_out)
        # predictions = inv_preds + inv_select[:, -1:] * null_pred  # (B, S)
        report["predictions"] = predictions
        # ---------------------------
        return predictions

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super(RuleLearner, self).get_config()
        config.update(
            {
                "inv_inputs": self.inv_inputs.numpy().tolist(),
                "inv_labels": self.inv_labels.numpy().tolist(),
                "num_symbols": self.num_symbols,
            }
        )
        return config


invs = [
    [1, 2, 4, 3, 4],
    [2, 3, 4, 5, 1],
    [3, 2, 4, 4, 7],
    [4, 1, 1, 2, 3],
    # [4, 2, 1, 2, 3],
    # [4, 3, 1, 2, 3],
    # [4, 5, 4, 4, 6],
    # [4, 5, 6, 2, 6],
    # [4, 2, 5, 6, 6],
]
# labels = [2, 3, 7, 1, 2, 3, 4, 6, 6]
labels = [2, 3, 7, 1]


def build_model() -> tf.keras.Model:
    """Build the trainable model."""
    seq_input = tf.keras.Input(
        shape=(1 + C["seq_length"],), name="input", dtype="int32"
    )
    predictions = RuleLearner(invs, labels, C["seq_symbols"])(seq_input)
    return tf.keras.Model(inputs=seq_input, outputs=predictions, name="rule_learner")
