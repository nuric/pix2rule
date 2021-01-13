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
parser = configlib.add_parser("Pix2Rule options.")
parser.add_argument(
    "--experiment_name",
    default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    help="Optional experiment name, default current datetime.",
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
    # Pre-compile debug run
    if C["debug"]:
        report = create_report(model, dsets["train"])
        print("Debug report keys:", report.keys())
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    model.summary()
    # ---
    run_name = run_name or utils.hashing.dict_hash(C)
    art_dir = Path(C["experiment_name"]) / run_name
    art_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Local artifact dir is %s", str(art_dir))
    # ---
    # Setup temperature scheduler callback
    temperature_callback = utils.callbacks.ParamScheduler(
        layer_name="object_selection",
        param_name="temperature",
        scheduler=tf.keras.optimizers.schedules.ExponentialDecay(
            0.5, decay_steps=1, decay_rate=0.9
        ),
        min_value=0.01,
    )
    # ---
    callbacks = [
        # tf.keras.callbacks.ModelCheckpoint(
        #     str(art_dir) + "/models/latest_model", monitor="loss"
        # ),
        temperature_callback,
        utils.callbacks.EarlyStopAtConvergence(C["converged_loss"]),
        utils.callbacks.TerminateOnNaN(),
        utils.callbacks.Evaluator(dsets),
        utils.callbacks.ArtifactSaver(dsets, art_dir),
    ]
    # ---------------------------
    # Training loop
    logger.info("Starting training.")
    model.fit(
        dsets["train"],
        epochs=C["max_steps"] // C["eval_every"],
        callbacks=callbacks,
        steps_per_epoch=C["eval_every"],
        verbose=0,
    )
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
    logger.info("Set experiment name %s", C["experiment_name"])
    mlflow.set_experiment(C["experiment_name"])
    # ---
    # Check for past run, are we resuming?
    logger.info("Searching past run with configuration hash %s", config_hash)
    run_id = None
    past_runs = mlflow.search_runs(
        filter_string=f"tags.config_hash = '{config_hash}'", max_results=1
    )
    if not past_runs.empty:
        # run_id = past_runs["run_id"][0] ***
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
    CONFIG_HASH = configlib.parse()
    print(f"Running with configuration hash {CONFIG_HASH}:")
    configlib.print_config()
    # ---------------------------
    mlflow_train()
