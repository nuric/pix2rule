"""Differentiable disjunctive normal form rule leanring component.

It really does not do anything remarkable besides implementing the tedious
operations that compute a learnable disjunctive normal form. It is naive in
the sense that it considers all permutations for variables bindings and does
not scale well with increasing number of objects. The learning part happens
by asking if each atom is in the rule or nor. For conjunctions we use, in the
rule, negation in the rule o not in the rule. For disjunctions, it is either
in the disjunction or not. The weights needs to be initialised carefully not
to starve the downstream layers of gradients otherwise it does not train at
all."""
from typing import Dict, List, Any
import itertools
import numpy as np
import tensorflow as tf

from components.ops import reduce_probsum, flatten_concat, soft_maximum, soft_minimum
from components.initialisers import CategoricalRandomNormal, BernoulliRandomNormal

configurable: Dict[str, Dict[str, Any]] = {
    "layer_name": {
        "type": str,
        "default": "DNF",
        "choices": ["DNF", "RealDNF"],
        "help": "DNF layer to use.",
    },
    "arities": {
        "type": int,
        "nargs": "*",
        "default": [0],
        "help": "Number of predicates and their arities.",
    },
    "num_total_variables": {
        "type": int,
        "default": 2,
        "help": "Number of variables in conjunctions.",
    },
    "num_conjuncts": {
        "type": int,
        "default": 8,
        "help": "Number of conjunctions in this DNF.",
    },
    "recursive": {
        "type": bool,
        "default": False,
        "help": "Whether the inputs contain layer outputs.",
    },
}


class DNF(tf.keras.layers.Layer):  # pylint: disable=too-many-instance-attributes
    """Single layer that represents conjunction with permutation invariance."""

    def __init__(
        self,
        arities: List[int] = None,
        num_total_variables: int = 2,
        num_conjuncts: int = 8,
        recursive: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # If we take material implication, number of conjuncts are the number of rules
        # Here we define the arity of these rules
        self.arities = arities or [0]
        # We use the following numbers to determine the structure of the rule
        # e.g. p(X) <- q(X) is 1 head and total 1
        assert (
            min(self.arities) >= 0 and max(self.arities) <= 2
        ), f"Arity of rules needs to be from 0 to 2, got {self.arities}."
        assert np.all(
            np.sort(self.arities) == self.arities
        ), f"Arity list needs to be sorted, got {self.arities}."
        assert (
            num_total_variables >= 0
        ), "Got negative number of total variables for DNF."
        self.num_total_variables = num_total_variables
        self.num_conjuncts = num_conjuncts
        self.recursive = recursive  # Tells if we should use outputs with given inputs
        self.temperature = self.add_weight(
            name="temperature",
            initializer=tf.keras.initializers.Constant(1),
            trainable=False,
        )
        # ---------------------------
        # Indices for THERE EXISTS operations if applicable
        try:
            binary_index = self.arities.index(2)
        except ValueError:
            binary_index = len(self.arities)
        try:
            unary_index = self.arities.index(1)
        except ValueError:
            unary_index = binary_index
        self.rule_idxs = (unary_index, binary_index)

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
            input_shape["nullary"][-1],
            input_shape["unary"][-1],
            input_shape["binary"][-1],
        )
        num_in = (
            pred0
            + self.num_total_variables * pred1
            + self.num_total_variables * (self.num_total_variables - 1) * pred2
        )
        # ---
        # pylint: disable=attribute-defined-outside-init
        # We will calculate a probability to have a stable start
        or_prob = 0.9
        and_prob = np.log(1 - np.power(0.5, 1 / (self.num_conjuncts * or_prob))) / (
            np.log(0.5) * num_in
        )
        assert (
            0 <= and_prob <= 1
        ), f"Conjunction element probability out of range: {and_prob}."
        self.and_kernel = self.add_weight(
            name="and_kernel",
            shape=(len(self.arities), self.num_conjuncts, num_in, 3),
            # initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            initializer=CategoricalRandomNormal(
                probs=[and_prob, 0.0, 1 - and_prob], mean=1.0, stddev=0.1
            )
            # regularizer=tf.keras.regularizers.L1(l1=0.01),
        )
        self.or_kernel = self.add_weight(
            name="or_kernel",
            shape=(len(self.arities), self.num_conjuncts),
            # initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            initializer=BernoulliRandomNormal(prob=or_prob, mean=1.0, stddev=0.1),
        )
        # ---------------------------
        # Compute permutation indices to gather later
        # For K permutations it tells what indices V variables take
        num_objects = input_shape["unary"][1]  # N
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

    def pad_inputs(self, inputs: Dict[str, tf.Tensor]):
        """Pad inputs to include layer outputs for recursive application."""
        # inputs {'nullary': (B, P0), 'unary': (B, N, P1),
        #         'binary': (B, N, N-1, P2)}
        keys = ["nullary", "unary", "binary"]
        counts = {k: self.arities.count(i) for i, k in enumerate(keys)}
        # We want to pad only the last dimension
        skip_dims = {k: [[0, 0]] * (i + 1) for i, k in enumerate(keys)}
        padded = {
            k: tf.pad(inputs[k], skip_dims[k] + [[0, counts[k]]], constant_values=0.0)
            for k in keys
        }
        # padded {'nullary_preds': (B, P0+R0), 'unary_preds': (B, N, P1+R1),
        #         'binary_preds': (B, N, N-1, P2+R2)}
        return padded

    def call(self, inputs: Dict[str, tf.Tensor], **kwargs):
        """Perform forward pass of the model."""
        # pylint: disable=too-many-locals
        # inputs {'nullary_preds': (B, P0), 'unary_preds': (B, N, P1),
        #         'binary_preds': (B, N, N-1, P2)}
        # ---------------------------
        # Setup permutations
        # Conjunctions in logic are permutation invariant A and B <=> B and A
        # so we take naive approach of evaluating all permutations since there is
        # no background knowledge / mode bias to restrict the search space.
        perm_nullary = tf.repeat(
            inputs["nullary"][:, None], self.perm_idxs.shape[0], axis=1
        )  # (B, K, P0)
        perm_unary = tf.gather(inputs["unary"], self.perm_idxs, axis=1)  # (B, K, V, P1)
        repeat_bidxs = tf.repeat(
            self.perm_bidxs[None], tf.shape(inputs["binary"])[0], axis=0
        )  # (B, K, V, V-1, 2)
        perm_binary = tf.gather_nd(
            inputs["binary"], repeat_bidxs, batch_dims=1
        )  # (B, K, V, V-1, P2)
        # ---------------------------
        # Compute flattened input
        in_tensor = flatten_concat(
            [perm_nullary, perm_unary, perm_binary], batch_dims=2
        )
        # (B, K, P0 + V*P1 + V*(V-1)*P2)
        # ---------------------------
        # Compute weighted conjunct truth values
        and_kernel = tf.nn.softmax(
            self.and_kernel / self.temperature, -1
        )  # (R, H, IN, 3)
        in_tensor = in_tensor[:, :, None, None]  # (B, K, 1, 1, IN)
        conjuncts_eval = (
            in_tensor * and_kernel[..., 0]
            + (1 - in_tensor) * and_kernel[..., 1]
            + and_kernel[..., 2]
        )  # (B, K, R, H, IN)
        # ---
        # AND operation
        conjuncts = tf.reduce_prod(conjuncts_eval, -1)  # (B, K, R, H)
        # ---------------------------
        # Compute weighted disjunct truth values
        or_kernel = tf.nn.sigmoid(self.or_kernel / self.temperature)  # (R, H)
        disjuncts = conjuncts * or_kernel  # (B, K, R, H)
        # ---
        # OR operation
        disjuncts = reduce_probsum(disjuncts, -1)  # (B, K, R)
        # ---------------------------
        # Reduce conjunction
        # We will reshape and reduce according to the arity of these conjuncts
        num_objects = tf.shape(inputs["unary"])[1]  # N
        object_perms = num_objects - tf.range(
            self.num_total_variables
        )  # [N, N-1, ...] x V
        eval_shape = tf.shape(disjuncts)  # [B, K, R]
        rule_shape = tf.concat([eval_shape[:1], object_perms, eval_shape[2:]], 0)
        rules = tf.reshape(disjuncts, rule_shape)  # (B, N, N-1, ..., R)
        # ---------------------------
        # For nullary reduce over all variables, i.e. there exists XYZ ...
        unary_index, binary_index = self.rule_idxs
        nullary_rules = rules[..., :unary_index]  # (B, N, N-1, ..., R0)
        var_range = tf.range(self.num_total_variables) + 1  # [1, 2, ...] x V
        nullary_rules = reduce_probsum(nullary_rules, var_range)  # (B, R0)
        # For unary reduce over remaining variables, i.e. For all X, there exists YZ
        unary_rules = rules[..., unary_index:binary_index]  # (B, N, N-1, ..., R1)
        unary_rules = reduce_probsum(unary_rules, var_range[1:])  # (B, N, R1)
        # For binary reduce over remaining variables except first 2 and so on
        binary_rules = rules[..., binary_index:]  # (B, N, N-1, ..., R2)
        binary_rules = reduce_probsum(binary_rules, var_range[2:])  # (B, N, N-1, R2)
        outputs = {
            "nullary": nullary_rules,
            "unary": unary_rules,
            "binary": binary_rules,
            "and_kernel": and_kernel,  # We return the kernels for analysis
            "or_kernel": or_kernel,
        }
        # ---------------------------
        # Merge or compute return values
        if self.recursive:
            keys = ["nullary", "unary", "binary"]
            # We assume the last R predicates are the learnt rules, so we slice
            # and concenate back their new values. i.e. amalgamate
            for i, k in enumerate(keys):
                count = self.arities.count(i)
                # Check if we have any variables at hand
                # tf.function should handle this if statement
                # as it is a fixed pure Python check
                if count == 0:
                    outputs[k] = inputs[k]
                    continue
                # old_value = inputs[k][..., -count:]  # (..., RX)
                # merged_value = tf.stack([old_value, outputs[k]], -1)  # (..., RX, 2)
                # Amalgamate function here, we use probabilistic sum
                # next_value = reduce_probsum(merged_value, -1)  # (..., RX)
                next_value = outputs[k]
                outputs[k] = tf.concat([inputs[k][..., :-count], next_value], -1)
        return outputs

    def get_config(self):
        """Serialisable configuration dictionary."""
        config = super().get_config()
        config.update(
            {
                "arities": self.arities,
                "num_total_variables": self.num_total_variables,
                "num_conjuncts": self.num_conjuncts,
                "recursive": self.recursive,
            }
        )
        return config


class RealDNF(DNF):
    """Real valued DNF layer."""

    def call(self, inputs: Dict[str, tf.Tensor], **kwargs):
        """Perform forward pass of the model."""
        # pylint: disable=too-many-locals
        # inputs {'nullary_preds': (B, P0), 'unary_preds': (B, N, P1),
        #         'binary_preds': (B, N, N-1, P2)}
        # ---------------------------
        # Setup permutations
        # Conjunctions in logic are permutation invariant A and B <=> B and A
        # so we take naive approach of evaluating all permutations since there is
        # no background knowledge / mode bias to restrict the search space.
        perm_nullary = tf.repeat(
            inputs["nullary"][:, None], self.perm_idxs.shape[0], axis=1
        )  # (B, K, P0)
        perm_unary = tf.gather(inputs["unary"], self.perm_idxs, axis=1)  # (B, K, V, P1)
        repeat_bidxs = tf.repeat(
            self.perm_bidxs[None], tf.shape(inputs["binary"])[0], axis=0
        )  # (B, K, V, V-1, 2)
        perm_binary = tf.gather_nd(
            inputs["binary"], repeat_bidxs, batch_dims=1
        )  # (B, K, V, V-1, P2)
        # ---------------------------
        # Compute flattened input
        in_tensor = flatten_concat(
            [perm_nullary, perm_unary, perm_binary], batch_dims=2
        )
        # (B, K, P0 + V*P1 + V*(V-1)*P2)
        # ---------------------------
        # Compute weighted conjunct truth values
        and_kernel = tf.nn.softmax(
            self.and_kernel / self.temperature, -1
        )  # (R, H, IN, 3)
        in_tensor = in_tensor[:, :, None, None]  # (B, K, 1, 1, IN)
        conjuncts_eval = (
            in_tensor * and_kernel[..., 0]
            + (in_tensor * -1) * and_kernel[..., 1]
            + and_kernel[..., 2]
            * soft_maximum(in_tensor, temperature=self.temperature, keepdims=True)
        )  # (B, K, R, H, IN)
        # ---
        # AND operation
        conjuncts = soft_minimum(conjuncts_eval, axis=-1, temperature=self.temperature)
        # (B, K, R, H)
        # ---------------------------
        # Compute weighted disjunct truth values
        or_kernel = tf.nn.sigmoid(self.or_kernel / self.temperature)  # (R, H)
        disjuncts = conjuncts * or_kernel + (1 - or_kernel) * soft_minimum(
            conjuncts, temperature=self.temperature, keepdims=True
        )
        # (B, K, R, H)
        # ---
        # OR operation
        disjuncts = soft_maximum(disjuncts, axis=-1, temperature=self.temperature)
        # (B, K, R)
        # ---------------------------
        # Reduce conjunction
        # We will reshape and reduce according to the arity of these conjuncts
        num_objects = tf.shape(inputs["unary"])[1:2]  # [N]
        # ---------------------------
        # For nullary reduce over all variables, i.e. there exists XYZ ...
        unary_index, binary_index = self.rule_idxs
        nullary_rules = disjuncts[..., :unary_index]  # (B, K, R0)
        nullary_rules = soft_maximum(
            nullary_rules, axis=1, temperature=self.temperature
        )  # (B, R0)
        # For unary reduce over remaining variables, i.e. For all X, there exists YZ
        unary_rules = disjuncts[..., unary_index:binary_index]  # (B, K, R1)
        unary_shape = tf.shape(unary_rules)  # [B, K, R1]
        unary_shape = tf.concat(
            [
                unary_shape[:1],
                num_objects,
                unary_shape[1:2] // num_objects,
                unary_shape[2:],
            ],
            0,
        )  # (B, N, K/N, R1)
        unary_rules = tf.reshape(unary_rules, unary_shape)
        unary_rules = soft_maximum(
            unary_rules, axis=2, temperature=self.temperature
        )  # (B, N, R1)
        # For binary reduce over remaining variables except first 2 and so on
        binary_rules = disjuncts[..., binary_index:]  # (B, K, R2)
        binary_shape = tf.shape(binary_rules)  # [B, K, R2]
        binary_shape = tf.concat(
            [
                binary_shape[:1],
                num_objects,
                num_objects - 1,
                binary_shape[1:2] // (num_objects * (num_objects - 1)),
                binary_shape[2:],
            ],
            0,
        )  # (B, N, N-1, K/(N*(N-1)), R2)
        binary_rules = tf.reshape(binary_rules, binary_shape)
        binary_rules = soft_maximum(
            binary_rules, axis=3, temperature=self.temperature
        )  # (B, N, N-1, R2)
        outputs = {
            "nullary": nullary_rules,
            "unary": unary_rules,
            "binary": binary_rules,
            "and_kernel": and_kernel,  # We return the kernels for analysis
            "or_kernel": or_kernel,
        }
        # ---------------------------
        # Merge or compute return values
        if self.recursive:
            keys = ["nullary", "unary", "binary"]
            # We assume the last R predicates are the learnt rules, so we slice
            # and concenate back their new values. i.e. amalgamate
            for i, k in enumerate(keys):
                count = self.arities.count(i)
                # Check if we have any variables at hand
                # tf.function should handle this if statement
                # as it is a fixed pure Python check
                if count == 0:
                    outputs[k] = inputs[k]
                    continue
                # old_value = inputs[k][..., -count:]  # (..., RX)
                # merged_value = tf.stack([old_value, outputs[k]], -1)  # (..., RX, 2)
                # Amalgamate function here, we use probabilistic sum
                # next_value = reduce_probsum(merged_value, -1)  # (..., RX)
                next_value = outputs[k]
                outputs[k] = tf.concat([inputs[k][..., :-count], next_value], -1)
        return outputs
