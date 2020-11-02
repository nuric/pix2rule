"""Differentiable rule learning component."""
from typing import Dict
import tensorflow as tf

from reportlib import report_tensor
from components import unification
from components import ops


class BaseRuleLearner(tf.keras.layers.Layer):
    """Base class for rule learning, handle predicates and unification."""

    def __init__(self, max_invariants: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.max_invariants = max_invariants  # upper bound on I

    def build(self, input_shape: Dict[str, tf.TensorShape]):
        """Build layer weights."""
        # pylint: disable=attribute-defined-outside-init
        # inputs {'unary_feats': (B, BL, P1), 'binary_feats': (B, BL, BL, P2),
        #         'inv_unary_feats': (I, IL, P1), 'inv_binary_feats': (I, IL, IL, P2),
        #         'inv_label': (I,)}
        ilen = self.max_invariants  # upper bound on I
        self.inv_unaryp = self.add_weight(
            name="inv_unaryp",
            shape=(ilen,) + input_shape["inv_unary_feats"][1:],
            initializer=tf.keras.initializers.RandomNormal(mean=-1),
            trainable=True,
        )  # (I, IL, IL, P1)
        self.inv_binaryp = self.add_weight(
            name="inv_binaryp",
            shape=(ilen,) + input_shape["inv_binary_feats"][1:],
            initializer=tf.keras.initializers.RandomNormal(mean=-1),
            trainable=True,
        )  # (I, IL, IL, P2)

    @staticmethod
    def softmax_scale(size: int, alpha: float = 0.999) -> tf.Tensor:
        """Compute the scale by which we need to multiply prior to softmax."""
        # alpha is softmax target scale on one-hot vector
        sm_scale = tf.math.log(
            alpha * tf.cast(size - 1, tf.float32) / (1 - alpha)
        )  # k = log(alpha * (n-1) / (1-alpha)) derived from softmax(kx)
        sm_scale = tf.cond(size == 1, lambda: 1.0, lambda: sm_scale)
        return sm_scale

    def call(self, inputs: Dict[str, tf.Tensor], **kwargs):
        """Perform forward pass of the model."""
        # pylint: disable=too-many-locals
        # inputs {'unary_feats': (B, BL, P1), 'binary_feats': (B, BL, BL, P2),
        #         'inv_unary_feats': (I, IL, P1), 'inv_binary_feats': (I, IL, IL, P2),
        #         'inv_label': (I,)}
        # ---------------------------
        num_invs = tf.shape(inputs["inv_unary_feats"])[0]  # I
        # Gather unary invariant map
        inv_unary_map = tf.nn.sigmoid(self.inv_unaryp[:num_invs])  # (I, IL, P1)
        # Gather binary invariant map
        inv_binary_map = tf.nn.sigmoid(self.inv_binaryp[:num_invs])  # (I, IL, IL, P2)
        # The examples need to satisfy these conditions
        inv_unary_conds = inv_unary_map * inputs["inv_unary_feats"]  # (I, IL, P1)
        inv_binary_conds = (
            inv_binary_map * inputs["inv_binary_feats"]
        )  # (I, IL, IL, P2)
        report_tensor("inv_unary_conds", inv_unary_conds)
        report_tensor("inv_binary_conds", inv_binary_conds)
        inv_loss = tf.reduce_sum(inv_unary_conds, [1, 2]) + tf.reduce_sum(
            inv_binary_conds, [1, 2, 3]
        )  # (I,)
        inv_loss = tf.reduce_mean(inv_loss) * 0.01  # ()
        self.add_loss(inv_loss)
        # ---------------------------
        # Unify
        uni_sets = unification.unify(
            inv_unary_conds,
            inv_binary_conds,
            inputs["unary_feats"],
            inputs["binary_feats"],
            iterations=1,
        )  # (B, I, IL, BL)
        report_tensor("uni_sets", uni_sets)
        # ---------------------------
        # Find which invariants unify
        inv_uni = tf.reduce_prod(ops.reduce_probsum(uni_sets, -1), -1)  # (B, I)
        report_tensor("inv_uni", inv_uni)
        # inv_uni = tf.concat([inv_uni, tf.ones((inv_uni.shape[0], 1))], -1)  # (B, I+1)
        # inv_select = leftright_cumprod(inv_uni)  # (B, I+1)
        # inv_select = tf.nn.softmax(tf.math.log(inv_uni / (1 - inv_uni)))  # (B, I)
        inv_select = tf.nn.softmax(inv_uni * self.softmax_scale(num_invs))  # (B, I)
        report_tensor("inv_select", inv_select)
        # ---------------------------
        return uni_sets, inv_select

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update({"max_invariants": self.max_invariants})
        return config


class SequencesRuleLearner(BaseRuleLearner):
    """Differentiable rule learning component for sequences dataset."""

    def build(self, input_shape: Dict[str, tf.TensorShape]):
        """Build layer weights."""
        # pylint: disable=attribute-defined-outside-init
        # inputs {'unary_feats': (B, BL, P1), 'binary_feats': (B, BL, BL, P2),
        #         'inv_unary_feats': (I, IL, P1), 'inv_binary_feats': (I, IL, IL, P2),
        #         'inv_label': (I,)}
        super().build(input_shape)
        ilen = self.max_invariants  # upper bound on I
        seq_len = input_shape["inv_unary_feats"][1]  # IL
        self.inv_out_map = self.add_weight(
            name="inv_out_map",
            shape=(ilen, seq_len + 1),  # +1 for constant output
            initializer="random_normal",
            trainable=True,
        )  # (I, IL+1)

    def call(self, inputs: Dict[str, tf.Tensor], **kwargs):
        """Perform forward pass of the model."""
        # pylint: disable=too-many-locals
        # inputs {'unary_feats': (B, BL, P1), 'binary_feats': (B, BL, BL, P2),
        #         'inv_unary_feats': (I, IL, P1), 'inv_binary_feats': (I, IL, IL, P2),
        #         'inv_label': (I,)}
        # ---------------------------
        # Compute which invariant to select / unify
        uni_sets, inv_select = super().call(inputs, **kwargs)  # (B, I, IL, BL), (B, I)
        # ---------------------------
        # Compute output edges
        num_nodes = tf.shape(inputs["unary_feats"])[1]  # BL
        sm_scale = self.softmax_scale(num_nodes)
        node_select = tf.nn.softmax(uni_sets * sm_scale, -1)  # (B, I, IL, BL)
        # (B, I, IL, BL) x (B, BL, S) -> (B, I, IL, S)
        edge_outs = tf.einsum(
            "bilk,bku->bilu", node_select, inputs["unary_feats"][..., 5:]
        )
        # ---------------------------
        # Select the output node
        # Either a node in the output, or the constant output node
        num_invs = tf.shape(inputs["inv_unary_feats"])[0]  # I
        inv_out_map = tf.nn.softmax(self.inv_out_map[:num_invs], -1)  # (I, IL+1)
        report_tensor("inv_out_map", inv_out_map)
        # (I, IL) x (B, I, IL, S) -> (B, I, S)
        inv_nodes_out = tf.einsum("il,bilp->bip", inv_out_map[:, :-1], edge_outs)
        inv_const_out = tf.one_hot(
            inputs["inv_label"],
            tf.shape(inv_nodes_out)[-1],
            on_value=1.0,
            off_value=0.0,
        )  # (I, S)
        inv_sym_out = inv_nodes_out + inv_out_map[:, -1:] * inv_const_out  # (B, I, S)
        report_tensor("inv_sym_out", inv_sym_out)
        # ---------------------------
        # (B, I) x (B, I, S) -> (B, S)
        predictions = tf.einsum("bi,bis->bs", inv_select, inv_sym_out)
        # inv_preds = tf.einsum("bi,bis->bs", inv_select[:, :-1], inv_sym_out)
        # predictions = inv_preds + inv_select[:, -1:] * null_pred  # (B, S)
        report_tensor("predictions", predictions)
        # ---------------------------
        return predictions


class RelsgameRuleLearner(BaseRuleLearner):
    """Relsgame rule learning component."""

    def call(self, inputs: Dict[str, tf.Tensor], **kwargs):
        """Perform forward pass of the model."""
        # pylint: disable=too-many-locals
        # inputs {'unary_feats': (B, BL, P1), 'binary_feats': (B, BL, BL, P2),
        #         'inv_unary_feats': (I, IL, P1), 'inv_binary_feats': (I, IL, IL, P2),
        #         'inv_label': (I,)}
        # ---------------------------
        # Compute which invariant to select / unify
        _, inv_select = super().call(inputs, **kwargs)  # (B, I, IL, BL), (B, I)
        # ---------------------------
        # Compute invariant label outputs
        inv_const_out = tf.one_hot(
            inputs["inv_label"],
            4,  # there are 4 labels in relsgame
            on_value=1.0,
            off_value=0.0,
        )  # (I, S)
        report_tensor("inv_sym_out", inv_const_out)
        # ---------------------------
        # (B, I) x (B, I, S) -> (B, S)
        predictions = tf.einsum("bi,is->bs", inv_select, inv_const_out)
        report_tensor("predictions", predictions)
        # ---------------------------
        return predictions
