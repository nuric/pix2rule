"""Differentiable rule learning component."""
from typing import Dict
import itertools
import numpy as np
import tensorflow as tf

from reportlib import report_tensor


class AndLayer(tf.keras.layers.Layer):
    """Single layer that represents conjunction with permutation invariance."""

    def __init__(
        self,
        num_conjuncts: int = 2,
        num_head_variables: int = 0,
        num_total_variables: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # If we take material implication, number of conjuncts are the number of rules
        self.num_conjuncts = num_conjuncts
        # We use the following numbers to determine the structure of the rule
        # e.g. p(X) <- q(X) is 1 head and total 1
        assert (
            num_head_variables >= 0
        ), "Got negative number of head variables for conjunct."
        assert (
            num_total_variables >= 0
        ), "Got negative number of total variables for conjunct."
        self.num_head_variables = num_head_variables
        self.num_total_variables = num_total_variables

    def build(self, input_shape: Dict[str, tf.TensorShape]):
        """Build layer weights."""
        # input_shape {'nullary_preds': (B, P0), 'unary_preds': (B, N, P1),
        #              'binary_preds': (B, N, N-1, P2)}
        # ---------------------------
        # The input is the flattened number of facts for the number of total variables
        # this conjunction contains
        # Flattened number of facts are the input
        # P0 + V*P1 + V*(V-1)*P2
        pred0, pred1, pred2 = (
            input_shape["nullary_preds"][-1],
            input_shape["unary_preds"][-1],
            input_shape["binary_preds"][-1],
        )
        num_in = (
            pred0
            + self.num_total_variables * pred1
            + self.num_total_variables * (self.num_total_variables - 1) * pred2
        )
        # ---
        # pylint: disable=attribute-defined-outside-init
        self.kernel = self.add_weight(
            name="kernel",
            shape=(num_in, self.num_conjuncts),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            regularizer=tf.keras.regularizers.L1(l1=0.01),
        )
        # ---------------------------
        # Compute permutation indices to gather later
        # For K permutations it tells what indices V variables take
        num_objects = input_shape["unary_preds"][1]  # N
        assert (
            self.num_total_variables <= num_objects
        ), f"More variables {self.num_total_variables} than objects {num_objects}"
        self.perm_idxs = np.array(
            list(itertools.permutations(range(num_objects), self.num_total_variables))
        )  # (K, V)
        # ---
        # Binary comparison indices for variables, XY XZ YX YZ ...
        binary_idxs = np.stack(
            np.nonzero(1 - np.eye(self.num_total_variables))
        ).T  # (V*(V-1), 2)
        binary_idxs = np.reshape(
            binary_idxs, (self.num_total_variables, self.num_total_variables - 1, 2)
        )  # (V, V-1, 2)
        perm_bidxs = self.perm_idxs[:, binary_idxs]  # (K, V, V-1, 2)
        # Due to omission of XX (the diagnoal) the indices on of second argument get shifted
        index_shift_mask = (perm_bidxs[..., 1] > perm_bidxs[..., 0]).astype(
            int
        )  # (K, V, V-1)
        self.perm_bidxs = np.stack(
            [perm_bidxs[..., 0], perm_bidxs[..., 1] - index_shift_mask], axis=-1
        )  # (K, V, V-1, 2)

    def call(self, inputs: Dict[str, tf.Tensor], **kwargs):
        """Perform forward pass of the model."""
        # pylint: disable=too-many-locals
        # inputs {'nullary_preds': (B, P0), 'unary_preds': (B, N, P1),
        #         'binary_preds': (B, N, N-1, P2)}
        # ---------------------------
        # Setup permutations
        # Conjunctions in logic are permutation invariant A and B = B and A
        # so we take naive approach of evaluating all permutations since there is
        # background knowledge to restrict the search space.
        perm_nullary = tf.repeat(
            inputs["nullary_preds"][:, None], self.perm_idxs.shape[0], axis=1
        )  # (B, K, P0)
        perm_unary = tf.gather(
            inputs["unary_preds"], self.perm_idxs, axis=1
        )  # (B, K, V, P1)
        repeat_bidxs = tf.repeat(
            self.perm_bidxs[None], tf.shape(inputs["binary_preds"])[0], axis=0
        )  # (B, K, V, V-1, 2)
        perm_binary = tf.gather_nd(
            inputs["binary_preds"], repeat_bidxs, batch_dims=1
        )  # (B, K, V, V-1, P2)
        # ---------------------------
        # Compute weighted conjunct truth values
        flattened_in = [
            tf.reshape(x, tf.concat([tf.shape(x)[:2], [-1]], 0))
            for x in (perm_unary, perm_binary)
        ]
        in_tensor: tf.Tensor = tf.concat(
            [perm_nullary] + flattened_in, -1
        )  # (B, K, P0 + V*P1 + V*(V-1)*P2)
        report_tensor("in_tensor", in_tensor)
        kernel = tf.nn.tanh(self.kernel)  # (IN, R)
        report_tensor("rule_kernel", kernel)
        weighted_truth = in_tensor[..., None] * kernel + (
            1 - tf.square(kernel)
        )  # (B, K, P0 + V*P1 + V*(V-1)*P2, R)
        # ---------------------------
        # Reduce conjunction
        conjuncts = tf.reduce_max(tf.reduce_min(weighted_truth, 2), 1)  # (B, R)
        return conjuncts
        # return inputs["nullary_preds"][:, :4]

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update({"num_conjuncts": self.num_conjuncts})
        return config


class BaseRuleLearner(
    tf.keras.layers.Layer
):  # pylint: disable=too-many-instance-attributes
    """Base class for rule learning, handle predicates and unification."""

    def __init__(
        self,
        max_invariants: int = 4,
        max_variables: int = 0,
        num_labels: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_invariants = max_invariants  # Upper bound on I
        # Number of of variables in rules, 0 to match input
        self.max_variables = max_variables
        self.num_labels = num_labels  # Number of labels / classes to predict
        self.perm_idxs = None  # Permutations indices

    def build(self, input_shape: Dict[str, tf.TensorShape]):
        """Build layer weights."""
        # pylint: disable=attribute-defined-outside-init
        # input_shape {'unary_feats': (B, BL, E), 'binary_feats': (B, BL, BL, E)}
        num_variables = self.max_variables or input_shape["unary_feats"][1]  # IL
        latent = input_shape["unary_feats"][2]  # E
        # ---------------------------
        # Model learnable parameters
        self.rule_unary = self.add_weight(
            name="rule_unary",
            shape=(self.max_invariants, num_variables, latent),
            initializer=tf.keras.initializers.RandomNormal(mean=0),
            trainable=True,
        )  # (I, IL, E)
        self.rule_binary = self.add_weight(
            name="rule_binary",
            shape=(self.max_invariants, num_variables, num_variables, latent),
            initializer=tf.keras.initializers.RandomNormal(mean=0),
            trainable=True,
        )  # (I, IL, IL, E)
        # ---
        # self.rule_labels = self.add_weight(
        # name="rule_labels",
        # shape=(self.max_invariants, self.num_labels),
        # initializer="random_normal",
        # trainable=True,
        # )  # (I, K)
        self.rule_labels = tf.one_hot([0, 0, 1, 1], 4, on_value=1.0, off_value=0.0)
        # ---------------------------
        # Permutation indices to gather later
        perm_idxs = tf.constant(
            list(
                itertools.permutations(
                    range(input_shape["unary_feats"][1]), num_variables
                )
            )
        )  # (P, IL)
        var_range = tf.broadcast_to(
            tf.range(num_variables), tf.shape(perm_idxs)
        )  # (P, IL)
        # ---
        self.perm_unary_idxs = tf.concat(
            [var_range[..., None], perm_idxs[..., None]], -1
        )  # (P, IL, 2)
        # ---
        edge_idxs = np.indices(
            (num_variables, num_variables), dtype=np.int32
        ).T  # (IL, IL, 2)
        perm_binary = tf.gather(perm_idxs, edge_idxs, axis=1)  # (P, IL, IL, 2)
        self.perm_binary_idxs = tf.concat(
            [tf.repeat(edge_idxs[None], perm_idxs.shape[0], axis=0), perm_binary], -1
        )  # (P, IL, IL, 4)

    @staticmethod
    def softmax_scale(size: int, alpha: float = 0.999) -> tf.Tensor:
        """Compute the scale by which we need to multiply prior to softmax."""
        # alpha is softmax target scale on one-hot vector
        sm_scale = tf.math.log(
            alpha * tf.cast(size - 1, tf.float32) / (1 - alpha)
        )  # k = log(alpha * (n-1) / (1-alpha)) derived from softmax(kx)
        sm_scale = tf.cond(tf.constant(size == 1), lambda: 1.0, lambda: sm_scale)
        return sm_scale

    def call(self, inputs: Dict[str, tf.Tensor], **kwargs):
        """Perform forward pass of the model."""
        # pylint: disable=too-many-locals
        # inputs {'unary_feats': (B, BL, E), 'binary_feats': (B, BL, BL, E)}
        # ---------------------------
        # Compute unary scores
        # We compute all pairwise combinations and then take permutations of them
        # to reduce the number of dot products to O(n^2) instead of O(n!)
        # (B, BL, E) x (I, IL, E) -> (B, I, IL, BL)
        unary_match = tf.einsum("ble,ike->bikl", inputs["unary_feats"], self.rule_unary)
        batch_size = tf.shape(inputs["unary_feats"])[0]  # B
        num_rules = tf.shape(self.rule_unary)[0]  # I
        batch_unary_idxs = tf.tile(
            self.perm_unary_idxs[None, None],
            [batch_size, num_rules, 1, 1, 1],
        )  # (B, I, P, IL, 2)
        unary_perm_scores = tf.gather_nd(
            unary_match, batch_unary_idxs, batch_dims=2
        )  # (B, I, P, IL)
        report_tensor("unary_perm_scores", unary_perm_scores)
        unary_scores = tf.reduce_sum(unary_perm_scores, -1)  # (B, I, P)
        # ---------------------------
        # Compute binary scores
        # (B, BL, BL, E) x (I, IL, IL, E) -> (B, I, IL, IL, BL, BL)
        binary_match = tf.einsum(
            "bjke,inme->binmjk", inputs["binary_feats"], self.rule_binary
        )
        batch_binary_idxs = tf.tile(
            self.perm_binary_idxs[None, None],
            [
                batch_size,
                num_rules,
                1,
                1,
                1,
                1,
            ],
        )  # (B, I, P, IL, IL, 4)
        binary_perm_scores = tf.gather_nd(
            binary_match, batch_binary_idxs, batch_dims=2
        )  # (B, I, P, IL, IL)
        report_tensor("binary_perm_scores", binary_perm_scores)
        binary_scores = tf.reduce_sum(binary_perm_scores, [-1, -2])  # (B, I, P)
        # ---------------------------
        # Merge unary and binary to obtain final score
        merged_scores = tf.reduce_min(unary_scores + binary_scores, -1)  # (B, I)
        inv_select = tf.nn.softmax(merged_scores, -1)  # (B, I)
        report_tensor("inv_select", inv_select)
        # ---------------------------
        # Compute label predictions
        report_tensor("rule_labels", self.rule_labels)
        predictions = inv_select @ self.rule_labels  # (B, K)
        return predictions

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
