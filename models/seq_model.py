"""Rule learning model for sequences dataset."""
import tensorflow as tf

from configlib import config as C
from reportlib import report
from components.sequence_features import SequenceFeatures
from .rule_learner import RuleLearner


def build_model() -> tf.keras.Model:
    """Build the trainable model."""
    # Setup inputs
    seq_len = 1 + C["seq_length"]  # 1+ for task_id
    seq_input = tf.keras.Input(shape=(seq_len,), name="input", dtype="int32")  # (B, BL)
    inv_input = tf.keras.Input(
        shape=(seq_len,), name="inv_input", dtype="int32"
    )  # (I, IL)
    inv_label = tf.keras.Input(shape=(), name="inv_label", dtype="int32")  # (I,)
    report["inv_input"] = inv_input
    report["inv_label"] = inv_label
    # Extract features
    feat_ext = SequenceFeatures()
    unary_feats, binary_feats = feat_ext(seq_input)  # (B, BL, P1), (B, BL, BL, P2)
    inv_unary_feats, inv_binary_feats = feat_ext(
        inv_input
    )  # (I, IL, P1), (I, IL, IL, P2)
    # Learn rules and predict
    combined = {
        "unary_feats": unary_feats,
        "binary_feats": binary_feats,
        "inv_unary_feats": inv_unary_feats,
        "inv_binary_feats": inv_binary_feats,
        "inv_label": inv_label,
    }
    predictions = RuleLearner(max_invariants=C["max_invariants"])(combined)  # (B, S)
    return tf.keras.Model(
        inputs=[seq_input, inv_input, inv_label],
        outputs=predictions,
        name="rule_learner",
    )
