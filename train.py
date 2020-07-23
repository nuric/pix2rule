"""Unification MLP."""
import os
import logging
import datetime
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import mlflow

import configlib
from configlib import config as C
from reportlib import create_report
import datasets
import models.rule_learner
import utils.callbacks

# Calm down tensorflow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel(logging.ERROR)
# Setup logging
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# Disable scientific printing
np.set_printoptions(suppress=True, precision=5, linewidth=180)

# ---------------------------

# Arguments
parser = configlib.add_parser("UMLP options.")
parser.add_argument("--experiment_name", help="Optional experiment name.")
parser.add_argument("--run_name", default="main", help="Optional experiment name.")
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


def train():
    """Training loop."""
    # Load data
    dsets = datasets.sequences.load_data()
    logger.info("Loaded datasets: %s", str(dsets))
    # ---------------------------
    # Setup model
    model = models.rule_learner.build_model()
    # Debug run
    if C["debug"]:
        report = create_report(model, dsets["train"])
        print("Debug report keys:", report.keys())
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    # ---------------------------
    # Setup Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            C["artifact_dir"] + "/models/latest_model", monitor="loss"
        ),
        utils.callbacks.EarlyStopAtConvergence(C["converged_loss"]),
        utils.callbacks.TerminateOnNaN(),
        utils.callbacks.Evaluator(dsets),
        utils.callbacks.ArtifactSaver(dsets),
    ]
    # ---------------------------
    # Training loop
    logger.info("Starting training.")
    model.fit(
        dsets["train"],
        epochs=C["max_steps"] // C["eval_every"],
        callbacks=callbacks,
        initial_epoch=0,
        steps_per_epoch=C["eval_every"],
        verbose=0,
    )
    # ---
    # Log post training artifacts
    logging.info("Training completed.")


def main():
    """Main entry point function."""
    # ---------------------------
    # Store in global config object inside configlib
    configlib.parse()
    print("Running with configuration:")
    configlib.print_config()
    # ---------------------------
    # Tensorflow graph mode (i.e. tf.function)
    tf.config.experimental_run_functions_eagerly(C["debug"])
    # ---------------------------
    # Setup MLflow
    if C["tracking_uri"]:
        mlflow.set_tracking_uri(C["tracking_uri"])
    logger.info("Tracking uri is %s", mlflow.get_tracking_uri())
    if not C["experiment_name"]:
        C["experiment_name"] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mlflow.set_experiment(C["experiment_name"])
    logger.info("Set experiment %s", C["experiment_name"])
    # ---
    art_dir = Path(C["experiment_name"]) / C["run_name"]
    art_dir.mkdir(parents=True, exist_ok=True)
    C["artifact_dir"] = str(art_dir)
    # ---------------------------
    # Big data machine learning in the cloud
    try:
        mlflow_run = mlflow.start_run()
        logger.info("Experiment id: %s", mlflow_run.info.experiment_id)
        logger.info("Run id: %s", mlflow_run.info.run_id)
        mlflow.log_params(C)
        mlflow.set_tag("config_hash", hash(json.dumps(C, sort_keys=True)))
        logger.info("Artifact uri is %s", mlflow.get_artifact_uri())
        train()
    except KeyboardInterrupt:
        mlflow.end_run(status="KILLED")
    else:
        mlflow.end_run()


if __name__ == "__main__":
    main()
