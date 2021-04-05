"""Training script for pix2rule."""
from typing import List, Dict
import logging
import datetime
from pathlib import Path
import re
import sys
import signal
import socket
import subprocess
import shutil

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
    # ---------------------------
    # Generate training file
    num_nullary = task_description["inputs"]["nullary"]["shape"][-1]  # num_nullary
    num_unary = task_description["inputs"]["unary"]["shape"][-1]  # num_unary
    num_binary = task_description["inputs"]["binary"]["shape"][-1]  # num_binary
    num_objects = task_description["inputs"]["unary"]["shape"][1]  # O
    num_variables = task_description["metadata"]["num_variables"]  # V
    # ---
    lines: List[str] = [
        "#modeh(t).",  # Our target we are learning, t :- ...
        f"#maxv({num_variables}).",  # Max number of variables.
        "#modeb(neq(var(obj), var(obj))).",  # Uniqueness of variables assumption.
        "neq(X, Y) :- obj(X), obj(Y), X != Y.",  # Definition of not equals.
        f"obj(0..{num_objects-1}).",  # Definition of objects, 0..n is inclusive so we subtract 1
        # These are bias constraints to adjust search space
        '#bias(":- body(neq(X, Y)), X >= Y.").',  # remove neg redundencies
        '#bias(":- body(naf(neq(X, Y))).").',  # We don't need not neg(..) in the rule
        '#bias(":- used(X), used(Y), X!=Y, not diff(X,Y).").',  # Make sure variable use is unique
        '#bias("diff(X,Y):- body(neq(X,Y)).").',
        '#bias("diff(X,Y):- body(neq(Y,X)).").',
        '#bias(":- body(neq(X, _)), not used(X).").',  # Make sure variable is used
        '#bias(":- body(neq(_, X)), not used(X).").',  # Make sure variable is used
        '#bias("used(X) :- body(unary(X, _)).").',  # We use a variable if it is in a unary
        '#bias("used(X) :- body(naf(unary(X, _))).").',
        '#bias("used(X) :- body(binary(X, _, _)).").',  # or a binary atom
        '#bias("used(X) :- body(binary(_, X, _)).").',
        '#bias("used(X) :- body(naf(binary(X, _, _))).").',
        '#bias("used(X) :- body(naf(binary(_, X, _))).").',
        '#bias(":- body(binary(X, X, _)).").',  # binary predicates are anti-reflexive
        '#bias(":- body(naf(binary(X, X, _))).").',
    ]
    # ---
    # Add nullary search space
    lines.append(f"#modeb({num_nullary}, nullary(const(nullary_type))).")
    # Add unary search space
    unary_size = num_variables * num_unary
    lines.append(f"#modeb({unary_size}, unary(var(obj), const(unary_type))).")
    # Add binary search space
    binary_size = num_variables * (num_variables - 1) * num_binary
    lines.append(
        f"#modeb({binary_size}, binary(var(obj), var(obj), const(binary_type)))."
    )
    # Add constants
    for ctype, count in [
        ("obj", num_objects),
        ("nullary_type", num_nullary),
        ("unary_type", num_unary),
        ("binary_type", num_binary),
    ]:
        for i in range(count):
            lines.append(f"#constant({ctype}, {i}).")
    # Add max penalty for ILASP
    total_size = num_nullary + unary_size + binary_size
    if C["train_type"] == "ilasp":
        lines.append(f"#max_penalty({total_size*2}).")
    # ---------------------------
    # Let's now generate and add examples
    logger.info("dataset: %s", str(task_description))
    examples = {**dsets["train"][0], **dsets["train"][1]}
    # {'nullary': (B, P0), 'unary': (B, O, P1), 'binary': (B, O, O-1, P2), 'label': (B,)}
    string_examples = utils.clingo.tensor_interpretations_to_strings(
        examples
    )  # list of lists
    logger.info("Generating %i examples.", len(string_examples))
    for i, (str_example, label) in enumerate(
        zip(string_examples, dsets["train"][1]["label"])
    ):
        # ---------------------------
        # Write the examples
        # We only have positive examples in which we either want t to be entailed
        # or not, and setup using inclusion and exclusions
        if label == 1:
            lines.append(f"#pos(eg{i}, {{t}}, {{}}, {{")
        else:
            lines.append(f"#pos(eg{i}, {{}}, {{t}}, {{")
        lines.extend(str_example)
        lines.append("}).")
    # ---------------------------
    # Save training file
    train_file = art_dir / "train.lp"
    with train_file.open("w") as fout:
        fout.writelines(f"{l}\n" for l in lines)
    with open("train_las.lp", "w") as fout:
        fout.writelines(f"{l}\n" for l in lines)
    # ---------------------------
    # Run the training, assuming ILASP and FastLAS in $PATH
    ilasp_cmd = [
        "ILASP",
        "--version=4",
        "--no-constraints",
        "--no-aggregates",
        f"-ml={total_size * 2}",  # We increase the size to allow for X!=Y etc in the rule
        f"--max-rule-length={total_size * 2}",
        "--strict-types",
        str(train_file),
    ]
    fastlas_cmd = ["FastLAS", str(train_file)]
    # Run command with a timeout of 1 hour
    timeout = 3600  # in seconds
    logger.info("Running symbolic learner with timeout %i.", timeout)
    try:
        res = subprocess.run(
            ilasp_cmd, capture_output=True, check=True, text=True, timeout=timeout
        )
        # res.stdout looks like this for ILASP:
        # t :- not nullary(0); unary(V1,0); obj(V1).
        # t :- not binary(V1,V2,0); not binary(V2,V1,0); ...

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # %% Pre-processing                          : 0.002s
        # %% Hypothesis Space Generation             : 0.053s
        # %% Conflict analysis                       : 1.789s
        # %%   - Positive Examples                   : 1.789s
        # %% Counterexample search                   : 0.197s
        # %%   - CDOEs                               : 0s
        # %%   - CDPIs                               : 0.196s
        # %% Hypothesis Search                       : 0.167s
        # %% Total                                   : 2.229s
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        with (art_dir / "train_cmd_out.txt").open("w") as fout:
            fout.write(res.stdout)
    except subprocess.TimeoutExpired:
        # We ran out of time
        logger.warning("Symbolic learner timed out.")
        learnt_rules: List[str] = list()  # [r1, r2] in string form
        total_time = float(timeout)
    else:
        res_lines = [l for l in res.stdout.split("\n") if l]
        comment_index = res_lines.index(
            r"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        )
        learnt_rules = res_lines[:comment_index]  # [r1, r2] in string form
        total_time = float(re.findall(r"\d.\d+", res_lines[-2])[0])
    logger.info("Learnt rules are %s", learnt_rules)
    logger.info("Total runtime was %f seconds.", total_time)
    # ---------------------------
    # Run the validation and test pipelines
    logger.info("Running validation.")
    learnt_program = learnt_rules + ["neq(X, Y) :- obj(X), obj(Y), X != Y."]
    with (art_dir / "learnt_program.lp").open("w") as fout:
        fout.write("\n".join(learnt_program))
    # ---------------------------
    report: Dict[str, float] = {"time": total_time}
    for key in dsets.keys():
        res = utils.clingo.clingo_rule_check(dsets[key][0], learnt_program)
        acc = np.mean(res == dsets[key][1]["label"])
        logger.info("%s accuracy is %f", key, acc)
        report[key + "_acc"] = acc
    # ---------------------------
    # Save artifacts to mlflow
    mlflow.log_artifacts(str(art_dir))
    shutil.rmtree(str(art_dir))
    mlflow.log_metrics(report, step=initial_epoch)


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
    # Setup artifacts and check if we are resuming
    run_name = run_name or utils.hashing.dict_hash(C)
    art_dir = Path(C["data_dir"]) / "active_runs" / C["experiment_name"] / run_name
    art_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Local artifact dir is %s", str(art_dir))
    # ---
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
        utils.callbacks.EarlyStopAtConvergence(delay=50),
        # tf.keras.callbacks.EarlyStopping(
        #     monitor="train_loss", min_delta=0.01, patience=10, verbose=1
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
