"""Unification MLP."""
import os
import logging
import numpy as np
import tensorflow as tf

import configlib
from configlib import config as C
import datasets
from models.rule_learner import RuleLearner

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
    model = RuleLearner()
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
