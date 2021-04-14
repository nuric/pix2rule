"""Training script for pix2rule."""
from typing import Dict
import logging
import datetime
from pathlib import Path
import sys
import signal
import socket
import shutil
import json

import numpy as np
import absl.logging
import tensorflow as tf
import mlflow
from mlflow.entities import RunStatus

import configlib
from configlib import config as C
from reportlib import create_report
import datasets
import models
import utils.callbacks
import utils.exceptions
import utils.hashing
import utils.clingo
import utils.ilasp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)

# Disable scientific printing
np.set_printoptions(suppress=True, precision=5, linewidth=180)

# ---------------------------

# Arguments
add_argument = configlib.add_group("Pix2Rule options", prefix="")
add_argument(
    "--experiment_name",
    default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    help="Optional experiment name, default current datetime.",
)
add_argument("--data_dir", default="data", help="Data folder.")
add_argument(
    "--train_type",
    default="deep",
    choices=["deep", "ilasp", "fastlas"],
    help="Type of model to train.",
)
add_argument(
    "--max_steps",
    default=4000,
    type=int,
    help="Maximum number of batch update steps.",
)
add_argument(
    "--eval_every", default=100, type=int, help="Evaluate model every N steps."
)
add_argument("--debug", action="store_true", help="Enable debug mode.")
add_argument("--tracking_uri", default="data/mlruns", help="MLflow tracking URI.")
add_argument(
    "--learning_rate", type=float, default=0.001, help="Optimizer learning rate."
)
add_argument(
    "--run_count",
    type=int,
    default=0,
    help="Run count for repeated runs of the same configuration.",
)

# ---------------------------


def train_ilp(run_name: str = None, initial_epoch: int = 0):
    """Train symbolic learners ILASP and FastLAS."""
    # ---------------------------
    # Load data
    task_description, dsets = datasets.get_dataset().load_data()
    logger.info("Loaded dataset: %s", str(task_description))
    # ---------------------------
    # Setup artifacts and check if we are resuming
    run_name = run_name or utils.hashing.dict_hash(C)
    art_dir = Path(C["data_dir"]) / "active_runs" / C["experiment_name"] / run_name
    art_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Local artifact dir is %s", str(art_dir))
    is_fastlas = C["train_type"] == "fastlas"
    # ---------------------------
    # Save task description obtained for this run
    task_desc_file = art_dir / "task_description.json"
    logger.info("Saving task description to %s", str(task_desc_file))
    with task_desc_file.open("w") as fout:
        json.dump(task_description, fout, indent=4)
    # ---------------------------
    # Generate search space
    # The following are lines of the LAS of file
    las_lines, max_size = utils.ilasp.generate_search_space(
        task_description, for_fastlas=is_fastlas
    )  # Mode biases etc.
    # ---------------------------
    # Let's now generate and add examples
    example_lines = utils.ilasp.generate_pos_examples(
        dsets["train"],
        with_noise=task_description["metadata"]["label_noise_probability"] > 0,
    )  # #pos({...}) examples
    # ---------------------------
    # Save training file
    all_lines = las_lines + example_lines
    train_file = art_dir / "train.lp"
    logger.info("Saving training file to: %s", str(train_file))
    with train_file.open("w") as fout:
        fout.writelines(f"{l}\n" for l in all_lines)
    # ---------------------------
    # Run the training, assuming ILASP and FastLAS in $PATH
    # Run command with a timeout of 1 hour
    timeout = 3600  # in seconds
    logger.info("Running symbolic learner with timeout %i.", timeout)
    # ---
    if is_fastlas:
        run_dict = utils.ilasp.run_fastlas(str(train_file), timeout=timeout)
    else:
        run_dict = utils.ilasp.run_ilasp(str(train_file), max_size, timeout=timeout)
    # ---
    with (art_dir / "train_cmd_out.txt").open("w") as fout:
        fout.write(run_dict["output"])
    logger.info("Learnt rules are %s", run_dict["learnt_rules"])
    logger.info("Total runtime was %f seconds.", run_dict["total_time"])
    # ---------------------------
    # Run the validation and test pipelines
    logger.info("Running validation.")
    learnt_program = run_dict["learnt_rules"] + ["neq(X, Y) :- obj(X), obj(Y), X != Y."]
    with (art_dir / "learnt_program.lp").open("w") as fout:
        fout.write("\n".join(learnt_program))
    # ---------------------------
    report: Dict[str, float] = {"time": run_dict["total_time"]}
    if "space_size" in run_dict:
        report["space_size"] = run_dict["space_size"]
    for key in dsets.keys():
        res = utils.clingo.clingo_rule_check(dsets[key][0], learnt_program)
        acc = np.mean(res == dsets[key][1]["label"])
        logger.info("%s accuracy is %f", key, acc)
        report[key + "_acc"] = acc
    # ---------------------------
    # Save artifacts to mlflow
    mlflow.log_artifacts(str(art_dir))
    shutil.rmtree(str(art_dir), ignore_errors=True)
    try:
        art_dir.parent.rmdir()  # Delete only if empty
    except OSError:
        pass  # we will keep the directory
    mlflow.log_metrics(report, step=initial_epoch)
    logger.info("Training completed.")


def train(run_name: str = None, initial_epoch: int = 0):
    """Training loop for single run."""
    # Load data
    task_description, dsets = datasets.get_dataset().load_data()
    logger.info("Loaded dataset: %s", str(task_description))
    # ---------------------------
    # Setup model
    model_dict = models.build_model(task_description)
    model = model_dict["model"]
    # Pre-compile debug run
    if C["debug"]:
        report = create_report(model, dsets["train"])
        print("Debug report keys:", report.keys())
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=C["learning_rate"]),
        loss=model_dict["loss"],
        metrics=model_dict["metrics"],
    )
    model_num_params = model.count_params()
    logger.info("Model has %i many parameters.", model.count_params())
    # ---------------------------
    # Setup artifacts
    run_name = run_name or utils.hashing.dict_hash(C)
    art_dir = Path(C["data_dir"]) / "active_runs" / C["experiment_name"] / run_name
    art_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Local artifact dir is %s", str(art_dir))
    # ---------------------------
    # Save task description obtained for this run
    task_desc_file = art_dir / "task_description.json"
    logger.info("Saving task description to %s", str(task_desc_file))
    with task_desc_file.open("w") as fout:
        json.dump(task_description, fout, indent=4)
    # ---------------------------
    # Are we resuming?
    saved_model_dir = art_dir / "models/latest_model"
    if saved_model_dir.exists():
        assert (
            initial_epoch > 0
        ), f"Expected initial to be greater than zero to resume training, got {initial_epoch}"
        # We are resuming
        logger.warning("Resuming training from %s", str(saved_model_dir))
        model = tf.keras.models.load_model(str(saved_model_dir))
        # Sanity check
        assert (
            model.count_params() == model_num_params
        ), f"Expected {model_num_params} but after resuming got {model.count_params()}!"
    # ---------------------------
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(saved_model_dir), monitor="validation_loss"
        ),
        utils.callbacks.TerminateOnNaN(),
        utils.callbacks.Evaluator(dsets),
        # utils.callbacks.EarlyStopAtConvergence(delay=50),
        # tf.keras.callbacks.EarlyStopping(
        #     monitor="validation_loss", min_delta=0.01, patience=10, verbose=1
        # ),
        utils.callbacks.DNFPruner(dsets, art_dir),
        utils.callbacks.ArtifactSaver(dsets, art_dir),
    ]
    # Merge in model callbacks if any
    if "callbacks" in model_dict:
        callbacks = model_dict["callbacks"] + callbacks
    # ---------------------------
    # Training loop
    logger.info("Starting training.")
    model.fit(
        dsets["train"],
        epochs=C["max_steps"] // C["eval_every"],
        callbacks=callbacks,
        initial_epoch=initial_epoch,
        steps_per_epoch=C["eval_every"],
        verbose=0,
    )
    # ---
    # Log post training artifacts
    logger.info("Training completed.")


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
    run_id, initial_epoch = None, 0
    past_runs = mlflow.search_runs(
        filter_string=f"tags.config_hash = '{config_hash}'", max_results=1
    )
    if not past_runs.empty:
        run_id = past_runs["run_id"][0]
        try:
            initial_epoch = int(past_runs["metrics.epoch"][0]) + 1
        except KeyError:
            initial_epoch = 0
        run_status = past_runs["status"][0]
        assert not RunStatus.is_terminated(
            RunStatus.from_string(run_status)
        ), f"Cannot resume a {run_status} run."
        logger.info("Should resume run with id %s from epoch %i", run_id, initial_epoch)
        mlflow.set_tag("resumed", True)
    # ---
    # Setup mlflow tracking
    mlflow_run = mlflow.start_run(run_id=run_id)
    mlflow.log_params(C)
    run_id = mlflow_run.info.run_id  # either brand new or the existing one
    logger.info("Experiment id: %s", mlflow_run.info.experiment_id)
    logger.info("Run id: %s", run_id)
    mlflow.set_tag("config_hash", config_hash)
    mlflow.set_tag("hostname", socket.gethostname())
    logger.info("Artifact uri is %s", mlflow.get_artifact_uri())
    # ---------------------------
    # Latch onto signal SIGTERM for graceful termination of long running
    # training jobs. Be nice to other people.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
    # The above code will raise SystemExit exception which we can catch
    # ---------------------------
    # Big data machine learning in the cloud
    status = RunStatus.FAILED
    try:
        tfunc = train if C["train_type"] == "deep" else train_ilp
        tfunc(run_name=run_id, initial_epoch=initial_epoch)
        status = RunStatus.FINISHED
    except KeyboardInterrupt:
        logger.warning("Killing training on keyboard interrupt.")
        status = RunStatus.KILLED
    except SystemExit:
        logger.warning("Pausing training on system exit.")
        status = RunStatus.SCHEDULED
    finally:
        mlflow.end_run(RunStatus.to_string(status))


if __name__ == "__main__":
    # ---------------------------
    # Store in global config object inside configlib
    CONFIG_HASH = configlib.parse()
    print(f"Running with configuration hash {CONFIG_HASH}:")
    configlib.print_config()
    # ---------------------------
    mlflow_train()
