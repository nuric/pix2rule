"""Differentiable disjunctive normal form rule learning component.

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
        "default": "WeightedDNF",
        "choices": ["DNF", "RealDNF", "WeightedDNF"],
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


class BaseDNF(tf.keras.layers.Layer):  # pylint: disable=too-many-instance-attributes
    """Single layer that represents conjunction with permutation invariance."""

    PRED_KEYS = ["nullary", "unary", "binary"]

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
        # pylint: disable=attribute-defined-outside-init
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
        # ---------------------------

    def pad_inputs(self, inputs: Dict[str, tf.Tensor], constant_values: float = 0.0):
        """Pad inputs to include layer outputs for recursive application."""
        # inputs {'nullary': (B, P0), 'unary': (B, N, P1),
        #         'binary': (B, N, N-1, P2)}
        counts = {k: self.arities.count(i) for i, k in enumerate(self.PRED_KEYS)}
        # We want to pad only the last dimension
        skip_dims = {k: [[0, 0]] * (i + 1) for i, k in enumerate(self.PRED_KEYS)}
        padded = {
            k: tf.pad(
                inputs[k],
                skip_dims[k] + [[0, counts[k]]],
                constant_values=constant_values,
            )
            for k in self.PRED_KEYS
        }
        # padded {'nullary_preds': (B, P0+R0), 'unary_preds': (B, N, P1+R1),
        #         'binary_preds': (B, N, N-1, P2+R2)}
        return padded

    def compute_permutations(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Compute permutations of given ground input facts."""
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
        return in_tensor

    def compute_conjunction(self, in_tensor: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Compute conjunction operation of given clauses."""
        # in_tensor (B, K, R, H, IN)
        raise NotImplementedError(f"BaseDNF must be inherited: {__name__}")

    def reduce_existential(self, tensor: tf.Tensor, axis: int) -> tf.Tensor:
        """Reduction operation for existential variables."""
        # tensor (B, K, N, ..., N-X, RX, H)
        raise NotImplementedError(f"BaseDNF must be inherited: {__name__}")

    def compute_disjunction(
        self, disjuncts: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """Compute disjunction of given clauses."""
        # disjuncts {'nullary': (B, R0), 'unary': (B, N, R1), 'binary': (B, N, N-1, R2)}
        # disjuncts {'nullary': (B, R0, H), 'unary': (B, N, R1, H), 'binary': (B, N, N-1, R2, H)}
        raise NotImplementedError(f"BaseDNF must be inherited: {__name__}")

    def apply_activation(  # pylint: disable=no-self-use
        self, rules: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """Apply an activation function to the final truth value of the rules."""
        # rules {'nullary': (B, R0), 'unary': (B, N, R1), 'binary': (B, N, N-1, R2)}
        return rules  # By default we don't do anything

    def call(
        self, inputs: Dict[str, tf.Tensor], **kwargs
    ):  # pylint: disable=too-many-locals
        """Perform forward pass of the model."""
        # inputs {'nullary_preds': (B, P0), 'unary_preds': (B, N, P1),
        #         'binary_preds': (B, N, N-1, P2)}
        # ---------------------------
        # Setup permutations
        in_tensor = self.compute_permutations(inputs)
        # (B, K, P0 + V*P1 + V*(V-1)*P2)
        # (B, K, IN)
        # ---------------------------
        # Compute conjunction
        conj_dict = self.compute_conjunction(in_tensor)
        assert "conjuncts" in conj_dict, f"conjuncts missing in {conj_dict.keys()}"
        # {'conjuncts': (B, K, R, H), ... other entries to log / analysis}
        conjuncts = conj_dict["conjuncts"]  # (B, K, R, H)
        # ---------------------------
        # Reduce conjunction
        # We will reshape and reduce according to the arity of these conjuncts
        num_objects = tf.shape(inputs["unary"])[1:2]  # [N]
        unary_index, binary_index = self.rule_idxs
        nullary_rules = conjuncts[..., :unary_index, :]  # (B, K, R0, H)
        nullary_rules = self.reduce_existential(nullary_rules, axis=1)  # (B, R0, H)
        # ---
        # For unary reduce over remaining variables, i.e. For all X, there exists YZ
        unary_rules = conjuncts[..., unary_index:binary_index, :]  # (B, K, R1, H)
        unary_shape = tf.shape(unary_rules)  # [B, K, R1, H]
        unary_shape = tf.concat(
            [
                unary_shape[:1],
                num_objects,
                unary_shape[1:2] // num_objects,
                unary_shape[2:],
            ],
            0,
        )  # (B, N, K/N, R1, H)
        unary_rules = tf.reshape(unary_rules, unary_shape)
        unary_rules = self.reduce_existential(unary_rules, axis=2)  # (B, N, R1, H)
        # ---
        # For binary reduce over remaining variables except first 2 and so on
        binary_rules = conjuncts[..., binary_index:, :]  # (B, K, R2, H)
        binary_shape = tf.shape(binary_rules)  # [B, K, R2, H]
        binary_shape = tf.concat(
            [
                binary_shape[:1],
                num_objects,
                num_objects - 1,
                binary_shape[1:2] // (num_objects * (num_objects - 1)),
                binary_shape[2:],
            ],
            0,
        )  # (B, N, N-1, K/(N*(N-1)), R2, H)
        binary_rules = tf.reshape(binary_rules, binary_shape)
        binary_rules = self.reduce_existential(
            binary_rules, axis=3
        )  # (B, N, N-1, R2, H)
        disjuncts = {
            "nullary": nullary_rules,
            "unary": unary_rules,
            "binary": binary_rules,
        }
        # ---------------------------
        # Compute disjunction
        rules_dict = self.compute_disjunction(disjuncts)
        assert set(self.PRED_KEYS).issubset(
            rules_dict.keys()
        ), f"Missing predicate keys in {rules_dict.keys()}"
        # {'nullary': (B, R0), 'unary': (B, N, R1), 'binary': (B, N, N-1, R2), ...}
        # ---------------------------
        # Optionally apply activation to the final value of the rules
        rules_dict = tf.cond(
            inputs.get("apply_activation", tf.constant(False)),
            lambda: self.apply_activation(rules_dict),
            lambda: rules_dict,
        )
        # ---------------------------
        # Collect all return elements
        rules_dict.update(conj_dict)
        # ---------------------------
        # Handle recursive case
        if self.recursive:
            # We assume the last R predicates are the learnt rules, so we slice
            # and concenate back their new values. i.e. amalgamate
            for i, k in enumerate(self.PRED_KEYS):
                count = self.arities.count(i)
                # Check if we have any variables at hand
                # tf.function should handle this if statement
                # as it is a fixed pure Python check
                if count == 0:
                    rules_dict[k] = inputs[k]
                    continue
                # old_value = inputs[k][..., -count:]  # (..., RX)
                # merged_value = tf.stack([old_value, outputs[k]], -1)  # (..., RX, 2)
                # Amalgamate function here, we use probabilistic sum
                # next_value = reduce_probsum(merged_value, -1)  # (..., RX)
                next_value = rules_dict[k]
                rules_dict[k] = tf.concat([inputs[k][..., :-count], next_value], -1)
        # ---------------------------
        return rules_dict

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


class DNF(BaseDNF):  # pylint: disable=too-many-instance-attributes
    """Single layer that represents conjunction with permutation invariance."""

    def build(self, input_shape: Dict[str, tf.TensorShape]):
        """Build layer weights."""
        # input_shape {'nullary_preds': (B, P0), 'unary_preds': (B, N, P1),
        #              'binary_preds': (B, N, N-1, P2)}
        # pylint: disable=attribute-defined-outside-init
        super().build(input_shape)
        # ---------------------------
        # The temperature parameter allows to tighting the sigmoid and softmax
        self.temperature = self.add_weight(
            name="temperature",
            initializer=tf.keras.initializers.Constant(1),
            trainable=False,
        )
        # ---------------------------
        # The input is the flattened number of facts for the number of total variables
        # this conjunction contains
        # Flattened number of facts are the input
        # P0 + V*P1 + V*(V-1)*P2
        pred0, pred1, pred2 = [input_shape[k][-1] for k in self.PRED_KEYS]
        num_in = (
            pred0
            + self.num_total_variables * pred1
            + self.num_total_variables * (self.num_total_variables - 1) * pred2
        )
        # ---
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
                probs=[and_prob / 2, and_prob / 2, 1 - and_prob], mean=1.0, stddev=0.1
            )
            # regularizer=tf.keras.regularizers.L1(l1=0.01),
        )
        self.or_kernel = self.add_weight(
            name="or_kernel",
            shape=(len(self.arities), self.num_conjuncts),
            # initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            initializer=BernoulliRandomNormal(prob=or_prob, mean=1.0, stddev=0.1),
        )

    def compute_conjunction(self, in_tensor: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Compute kernel based conjunction."""
        # in_tensor (B, K, R, H, IN)
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
        # ---------------------------
        # AND operation
        conjuncts = tf.reduce_prod(conjuncts_eval, -1)  # (B, K, R, H)
        # ---------------------------
        return {"conjuncts": conjuncts, "and_kernel": and_kernel}

    def reduce_existential(self, tensor: tf.Tensor, axis: int) -> tf.Tensor:
        """Reduction operation for existential variables."""
        # tensor (B, K, N, ..., N-X, RX, H)
        return reduce_probsum(tensor, axis=axis)

    def compute_disjunction(
        self, disjuncts: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """Compute disjunction of given clauses."""
        # disjuncts {'nullary': (B, R0, H), 'unary': (B, N, R1, H), 'binary': (B, N, N-1, R2, H)}
        # ---------------------------
        # Setup disjunction kernel
        or_kernel = tf.nn.sigmoid(self.or_kernel / self.temperature)  # (R, H)
        unary_index, binary_index = self.rule_idxs
        nullary_or_kernel = or_kernel[:unary_index]  # (R0, H)
        unary_or_kernel = or_kernel[unary_index:binary_index]  # (R1, H)
        binary_or_kernel = or_kernel[binary_index:]  # (R2, H)
        kernels = {
            "nullary": nullary_or_kernel,
            "unary": unary_or_kernel,
            "binary": binary_or_kernel,
        }
        # ---------------------------
        # OR operation
        rules = {k: reduce_probsum(v * kernels[k], -1) for k, v in disjuncts.items()}
        # {'nullary': (B, R0), 'unary': (B, N, R1), 'binary': (B, N, N-1, R2)}
        # ---------------------------
        rules["or_kernel"] = or_kernel  # for logging and analysis
        return rules


class RealDNF(DNF):  # pylint: disable=too-many-ancestors
    """Real valued DNF layer."""

    def compute_conjunction(self, in_tensor: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Compute kernel based conjunction."""
        # in_tensor (B, K, R, H, IN)
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
        return {"conjuncts": conjuncts, "and_kernel": and_kernel}

    def reduce_existential(self, tensor: tf.Tensor, axis: int) -> tf.Tensor:
        """Reduction operation for existential variables."""
        # tensor (B, K, N, ..., N-X, RX, H)
        return soft_maximum(tensor, axis=axis, temperature=self.temperature)

    def compute_disjunction(
        self, disjuncts: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """Compute disjunction of given clauses."""
        # disjuncts {'nullary': (B, R0, H), 'unary': (B, N, R1, H), 'binary': (B, N, N-1, R2, H)}
        # ---------------------------
        # Setup disjunction kernel
        or_kernel = tf.nn.sigmoid(self.or_kernel / self.temperature)  # (R, H)
        unary_index, binary_index = self.rule_idxs
        nullary_or_kernel = or_kernel[:unary_index]  # (R0, H)
        unary_or_kernel = or_kernel[unary_index:binary_index]  # (R1, H)
        binary_or_kernel = or_kernel[binary_index:]  # (R2, H)
        kernels = {
            "nullary": nullary_or_kernel,
            "unary": unary_or_kernel,
            "binary": binary_or_kernel,
        }
        # ---------------------------
        # OR operation
        rules = {
            k: soft_maximum(
                v * kernels[k]
                + (1 - kernels[k])
                * soft_minimum(v, temperature=self.temperature, keepdims=True),
                axis=-1,
                temperature=self.temperature,
            )
            for k, v in disjuncts.items()
        }
        # {'nullary': (B, R0), 'unary': (B, N, R1), 'binary': (B, N, N-1, R2)}
        # ---------------------------
        rules["or_kernel"] = or_kernel  # for logging and analysis
        return rules


def fixed_weight_disjunction(
    tensor: tf.Tensor, axis: int = -1, fixed_weight: float = 4.0
) -> tf.Tensor:
    """Compute a disjunction using the fixed weight."""
    # tensor (..., X, ...)
    in_num = tf.cast(tf.shape(tensor)[axis], tf.float32)  # X
    in_magnitude = tf.reduce_mean(tf.math.abs(tensor), axis)  # (..., ...)
    fixed_bias = (in_num - 1) * fixed_weight * in_magnitude  # ()
    or_tensor = tf.reduce_sum(tensor * fixed_weight, axis)  # (..., ...)
    return or_tensor + fixed_bias


class WeightedDNF(BaseDNF):  # pylint: disable=too-many-instance-attributes
    """Single layer that represents DNF with permutation invariance."""

    def build(self, input_shape: Dict[str, tf.TensorShape]):
        """Build layer weights."""
        # input_shape {'nullary_preds': (B, P0), 'unary_preds': (B, N, P1),
        #              'binary_preds': (B, N, N-1, P2)}
        super().build(input_shape)
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
        self.and_kernel = self.add_weight(
            name="and_kernel",
            shape=(len(self.arities), self.num_conjuncts, num_in),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            # regularizer=tf.keras.regularizers.L1(l1=0.01),
        )
        self.or_kernel = self.add_weight(
            name="or_kernel",
            shape=(len(self.arities), self.num_conjuncts),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
        )
        self.success_threshold = self.add_weight(
            name="success_threshold",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.2),
            constraint=tf.keras.constraints.NonNeg(),
            trainable=False,
        )
        # ---------------------------

    def compute_conjunction(self, in_tensor: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Compute kernel based conjunction."""
        # in_tensor (B, K, IN)
        # ---------------------------
        tanh_and_kernel = tf.nn.tanh(self.and_kernel)  # (R, H, IN)
        and_weights = (
            tanh_and_kernel
            * self.success_threshold
            / tf.nn.tanh(self.success_threshold)
        )  # (R, H, IN)
        conjuncts_eval = in_tensor[:, :, None, None] * and_weights  # (B, K, R, H, IN)
        and_wsum = 1 - tf.reduce_sum(tf.math.abs(tanh_and_kernel), -1)  # (R, H)
        and_bias = self.success_threshold * and_wsum  # (R, H)
        conjuncts = tf.reduce_sum(conjuncts_eval, -1) + and_bias  # (B, K, R, H)
        conjuncts = tf.nn.tanh(conjuncts)  # (B, K, R, H)
        # ---------------------------
        return {"conjuncts": conjuncts, "and_kernel": self.and_kernel}

    def reduce_existential(self, tensor: tf.Tensor, axis: int) -> tf.Tensor:
        """Reduction operation for existential variables."""
        # tensor (B, K, N, ..., N-X, RX, H)
        return tf.reduce_max(tensor, axis=axis)

    def compute_disjunction(  # pylint: disable=too-many-locals
        self, disjuncts: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """Compute disjunction of given clauses."""
        # disjuncts {'nullary': (B, R0, H), 'unary': (B, N, R1, H), 'binary': (B, N, N-1, R2, H)}
        # ---------------------------
        # Setup disjunction kernel
        unary_index, binary_index = self.rule_idxs
        nullary_or_kernel = self.or_kernel[:unary_index]  # (R0, H)
        unary_or_kernel = self.or_kernel[unary_index:binary_index]  # (R1, H)
        binary_or_kernel = self.or_kernel[binary_index:]  # (R2, H)
        kernels = {
            "nullary": nullary_or_kernel,
            "unary": unary_or_kernel,
            "binary": binary_or_kernel,
        }
        # ---------------------------
        # OR operation
        rules: Dict[str, tf.Tensor] = dict()
        # threshold of success, maximum we expect to output
        or_threshold = self.success_threshold
        for k, kernel in kernels.items():
            tanh_kernel = tf.nn.tanh(kernel)  # (RX, H)
            or_weights = (
                tanh_kernel
                * self.success_threshold
                / tf.nn.tanh(self.success_threshold)
            )  # (RX, H)
            kernel_sum = tf.reduce_sum(tf.math.abs(tanh_kernel), -1) - 1  # (RX,)
            disj_bias = kernel_sum * or_threshold  # (RX,)
            disj = (
                tf.reduce_sum(disjuncts[k] * or_weights, -1) + disj_bias
            )  # (B, ..., RX)
            rules[k] = disj
        # {'nullary': (B, R0), 'unary': (B, N, R1), 'binary': (B, N, N-1, R2)}
        # ---------------------------
        rules["or_kernel"] = self.or_kernel  # for logging and analysis
        return rules

    def apply_activation(self, rules: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Apply an activation function to the final truth value of the rules."""
        # rules {'nullary': (B, R0), 'unary': (B, N, R1), 'binary': (B, N, N-1, R2)}
        # Apply a tanh activation for weighted logic
        return {
            k: tf.nn.tanh(v) if k in self.PRED_KEYS else v for k, v in rules.items()
        }
