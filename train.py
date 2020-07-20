"""Unification MLP."""
import os
import logging
import datetime
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import mlflow

import configlib
from configlib import config as C
import datasets
from models.rule_learner import RuleLearner

# Calm down tensorflow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Setup logging
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# Disable scientific printing
np.set_printoptions(suppress=True, precision=5, linewidth=180)

# ---------------------------

# Arguments
parser = configlib.add_parser("UMLP options.")
parser.add_argument(
    "--invariants", default=1, type=int, help="Number of invariants per task."
)
parser.add_argument(
    "--max_steps", default=4000, type=int, help="Maximum number of batch update steps.",
)
parser.add_argument(
    "--converged_loss",
    default=0.001,
    type=float,
    help="Loss below which convergence is achieved.",
)
parser.add_argument(
    "--eval_every", default=100, type=int, help="Evaluate model every N steps."
)
parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
parser.add_argument("--tracking_uri", help="MLflow tracking URI.")

# ---------------------------


def train_step(model, batch, lossf, optimiser):
    """Perform one batch update."""
    # batch = {'input': (B, 1+L), 'label': (B,)}
    report = dict()
    with tf.GradientTape() as tape:
        report = model(batch["input"], training=True)  # {'predictions': (B, S), ...}
        loss = lossf(batch["label"], report["predictions"])  # (B, I)
        loss += sum(model.losses)  # Keras accumulated losses, e.g. regularisers
        report["loss"] = loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimiser.apply_gradients(zip(gradients, model.trainable_variables))
    return report


def training_loop(model, dsets, metrics):
    """The main training loop for training evaluating a model."""
    # ---------------------------
    lossf = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimiser = tf.keras.optimizers.Adam()
    # ---------------------------
    for i, batch in dsets["train"].enumerate():
        ltime = time.time()
        step = i.numpy()
        # batch = {'input': (B, 1+L), 'label': (B,)}
        # batch = {"input": tf.constant([[1, 2, 4, 3, 4]]), "label": tf.constant([2])}
        report = train_step(model, batch, lossf, optimiser)
        # ---
        # Check for NaN
        if tf.math.is_nan(report["loss"]):
            logger.critical("Loss is NaN.")
            mlflow.set_tag("result", "NaN")
            break
        # ---
        # Training metric collection
        metrics["train"]["total_loss"].update_state(report["loss"])
        metrics["train"]["loss"].update_state(batch["label"], report["predictions"])
        metrics["train"]["acc"].update_state(batch["label"], report["predictions"])
        # Run evaluation and metric collection
        if step % C["eval_every"] == 0:
            eval_model(model, dsets, metrics)
            flat_metrics = {
                dkey + "_" + mkey: metric.result().numpy()
                for dkey, dmetrics in metrics.items()
                for mkey, metric in dmetrics.items()
            }
            flat_metrics["time"] = time.time() - ltime
            print(
                "Step:",
                step,
                " ".join(
                    [k + " " + "{:.3f}".format(v) for k, v in flat_metrics.items()]
                ),
            )
            mlflow.log_metrics(flat_metrics, step=step)
        # ---
        # Check terminating conditions
        if step == C["max_steps"]:
            logger.info("Completed max number of steps.")
            mlflow.set_tag("result", "complete")
            break
        if report["loss"].numpy() < C["converged_loss"]:
            logger.info("Training converged.")
            mlflow.set_tag("result", "converged")
            break


def eval_model(model, dsets, metrics):
    """Evaluate given model on test datasets updating metrics."""
    for k, dset in dsets.items():
        if not k.startswith("test"):
            continue
        for batch in dset:
            report = model(
                batch["input"], training=False
            )  # {'predictions':, (B, S), ...}
            metrics[k]["loss"].update_state(batch["label"], report["predictions"])
            metrics[k]["acc"].update_state(batch["label"], report["predictions"])


def artifact_model(model, dsets):
    """Run and log post run artifacts of the model."""
    # Check and create artifacts directory
    Path("artifacts").mkdir(exist_ok=True)
    # ---
    art_count = 0
    for k, dset in dsets.items():
        if not k.startswith("test"):
            continue
        for batch in dset.take(1):
            report = model(
                batch["input"], training=False
            )  # {'predictions': (B, S), ...}
            report = {k: v.numpy() for k, v in report.items()}
            np.savez_compressed("artifacts/" + k + "_report.npz", **report)
            art_count += 1
    # ---
    logger.info("Saved %i many artifacts.", art_count)
    # TODO(nuric): save model weights
    # ---
    mlflow.log_artifacts("artifacts")


def train():
    """Training loop."""
    # Load data
    dsets = datasets.sequences.load_data()
    logger.info("Loaded datasets: %s", str(dsets))
    # ---------------------------
    # Setup model
    model = RuleLearner()
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    # ---------------------------
    # Setup Callbacks
    # ---------------------------
    # Training loop
    logger.info("Starting training.")
    try:
        model.fit(
            dsets["train"],
            epochs=C["max_steps"] // C["eval_every"],
            callbacks=None,
            initial_epoch=0,
            steps_per_epoch=C["eval_every"],
        )
    except KeyboardInterrupt:
        logger.warning("Early stopping due to KeyboardInterrupt.")
    # ---
    # Log post training artifacts
    logging.info("Training completed.")
    # artifact_model(model, dsets)
    logging.info("Artifacts saved.")


def main():
    """Main entry point function."""
    # ---------------------------
    # Store in global config object inside configlib
    parsed_conf = configlib.parse()
    print("Running with configuration:")
    configlib.print_config()
    # ---------------------------
    # Tensorflow graph mode (i.e. tf.function)
    tf.config.experimental_run_functions_eagerly(parsed_conf["debug"])
    # ---------------------------
    # Setup MLflow
    if parsed_conf["tracking_uri"]:
        mlflow.set_tracking_uri(parsed_conf["tracking_uri"])
    logger.info("Tracking uri is %s", mlflow.get_tracking_uri())
    experiment_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mlflow.set_experiment(experiment_name)
    logger.info("Created experiment %s", experiment_name)
    # ---------------------------
    # Big data machine learning in the cloud
    with mlflow.start_run():
        mlflow.log_params(parsed_conf)
        logger.info("Artifact uri is %s", mlflow.get_artifact_uri())
        train()


if __name__ == "__main__":
    main()
