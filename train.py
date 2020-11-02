"""Unification MLP."""
import logging
import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import mlflow

import configlib
from configlib import config as C
from reportlib import create_report
import datasets
import models
import utils.callbacks
import utils.exceptions
import utils.hashing

# Setup logging
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)

# Disable scientific printing
np.set_printoptions(suppress=True, precision=5, linewidth=180)

# ---------------------------

# Arguments
parser = configlib.add_parser("UMLP options.")
parser.add_argument("--experiment_name", help="Optional experiment name.")
parser.add_argument(
    "--max_invariants", default=4, type=int, help="Number of maximum invariants."
)
parser.add_argument(
    "--max_steps",
    default=4000,
    type=int,
    help="Maximum number of batch update steps.",
)
parser.add_argument(
    "--converged_loss",
    default=0.01,
    type=float,
    help="Loss below which convergence is achieved.",
)
parser.add_argument(
    "--eval_every", default=100, type=int, help="Evaluate model every N steps."
)
parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
parser.add_argument("--tracking_uri", help="MLflow tracking URI.")

# ---------------------------


def train(run_name: str = None):
    """Training loop for single run."""
    # Load data
    dsets = datasets.load_data()
    logger.info("Loaded datasets: %s", str(dsets))
    # ---------------------------
    # Setup model
    model = models.build_model()
    # ---------------------------
    # Setup Callbacks
    inv_selector = utils.callbacks.InvariantSelector(
        dsets["train"], max_invariants=C["max_invariants"]
    )
    # ---
    # Pre-compile debug run
    if C["debug"]:
        report = create_report(model, inv_selector.create_dataset())
        print("Debug report keys:", report.keys())
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    # ---
    run_name = run_name or utils.hashing.dict_hash(C)
    art_dir = Path(C["experiment_name"]) / run_name
    art_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Local artifact dir is %s", str(art_dir))
    callbacks = [
        inv_selector,
        tf.keras.callbacks.ModelCheckpoint(
            str(art_dir) + "/models/latest_model", monitor="loss"
        ),
        utils.callbacks.EarlyStopAtConvergence(C["converged_loss"]),
        utils.callbacks.TerminateOnNaN(),
        utils.callbacks.Evaluator(dsets, inv_selector.create_dataset),
        utils.callbacks.ArtifactSaver(dsets, art_dir, inv_selector.create_dataset),
    ]
    # ---------------------------
    # Training loop
    logger.info("Starting training.")
    while True:
        try:
            model.fit(
                inv_selector.create_dataset(),
                epochs=C["max_steps"] // C["eval_every"],
                callbacks=callbacks,
                initial_epoch=inv_selector.last_epoch,
                steps_per_epoch=C["eval_every"],
                verbose=0,
            )
        except utils.exceptions.NewInvariantException:
            logger.info("Resuming training with new invariants.")
            # print("Invs:", inv_selector.inv_inputs, sep="\n")
        else:
            break
    # ---
    # Log post training artifacts
    logging.info("Training completed.")


def mlflow_train():
    """Setup mlflow and train."""
    # ---------------------------
    # Curate configuration parameters
    config_hash = utils.hashing.dict_hash(C)
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
    # Check for past run, are we resuming?
    logger.info("Configuration hash is %s", config_hash)
    run_id = None
    past_runs = mlflow.search_runs(
        filter_string=f"tags.config_hash = '{config_hash}'", max_results=1
    )
    if not past_runs.empty:
        # run_id = past_runs["run_id"][0]
        logger.info("Resuming run with id %s", run_id)
        raise NotImplementedError("Resuming runs in the same experiment.")
    # ---
    # Setup mlflow tracking
    mlflow_run = mlflow.start_run(run_id=run_id)
    mlflow.log_params(C)
    run_id = mlflow_run.info.run_id  # either brand new or the existing one
    logger.info("Experiment id: %s", mlflow_run.info.experiment_id)
    logger.info("Run id: %s", run_id)
    mlflow.set_tag("config_hash", config_hash)
    logger.info("Artifact uri is %s", mlflow.get_artifact_uri())
    # ---------------------------
    # Big data machine learning in the cloud
    try:
        train(run_name=run_id)
    except KeyboardInterrupt:
        logger.warning("Pausing training on keyboard interrupt.")
    else:
        mlflow.end_run()


if __name__ == "__main__":
    # ---------------------------
    # Store in global config object inside configlib
    configuration_hash = configlib.parse()
    print("Running with configuration hash {configuration_hash}:")
    configlib.print_config()
    # ---------------------------
    mlflow_train()
