"""Custom training utility callbacks."""
from typing import List, Dict, Callable, Tuple
import shutil
from pathlib import Path
import time

import numpy as np
import tensorflow as tf
import mlflow
import mlflow.keras

import models
from reportlib import create_report
from . import exceptions


class InvariantSelector(
    tf.keras.callbacks.Callback
):  # pylint: disable=too-many-instance-attributes
    """Selects new invariants and updates model invariants."""

    def __init__(
        self, dataset: tf.data.Dataset, max_invariants: int = 4, patience: int = 10
    ):
        super().__init__()
        self.dataset = dataset
        # Start with null invariant to get a forward pass from model
        inputs_spec: Dict[str, tf.TensorSpec]
        label_spec: tf.TensorSpec
        # ({'key': TensorSpec, ...}, TensorSpec)
        inputs_spec, label_spec = dataset.element_spec
        self.inv_inputs = {
            k: np.zeros((1,) + v.shape[1:], dtype=v.dtype.as_numpy_dtype)
            for k, v in inputs_spec.items()
        }
        self.inv_label = np.zeros(
            (1,) + label_spec.shape[1:], dtype=label_spec.dtype.as_numpy_dtype
        )
        self.is_blank_inv = True  # Flag to indicate if invariants are zeros
        self.patience = patience
        self.max_invariants = max_invariants
        # Instance variables
        self.wait = 0
        self.best = np.inf
        self.last_epoch = 0

    def create_dataset(self, origin: tf.data.Dataset = None) -> tf.data.Dataset:
        """Create new dataset with invariant as inputs."""

        def add_invariants(inputs: Dict[str, tf.Tensor], label: tf.Tensor):
            """Add invariants to inputs tensor dictionary."""
            inputs.update({"inv_" + k: v for k, v in self.inv_inputs.items()})
            inputs["inv_label"] = self.inv_label
            return inputs, label

        return (
            origin.map(add_invariants)
            if origin is not None
            else self.dataset.map(add_invariants)
        )

    def on_train_begin(self, logs: Dict[str, float] = None):
        """Select invariants at the beginning of training."""
        # Get a random batch example
        if not self.is_blank_inv:
            return
        report = create_report(self.model, self.create_dataset())
        ridxs = np.random.choice(len(report["label"]), size=10, replace=False)
        # ---
        # _, ridxs = np.unique(report["label"], return_index=True)
        # ---
        self.inv_inputs = {k: report[k][ridxs] for k in self.dataset.element_spec[0]}
        self.inv_label = report["label"][ridxs]
        print("Starting with invariant inputs with labels:", self.inv_label)
        # Signal to update training dataset
        self.is_blank_inv = False
        raise exceptions.NewInvariantException()

    def on_epoch_end(self, epoch, logs: Dict[str, float] = None):
        """Check and add one more invariant."""
        # check for stagnation
        self.wait += 1
        logs = logs or {}
        current = logs.get("loss", self.best)
        self.last_epoch = epoch
        if (self.best - current) > 0.1:
            self.wait = 0
            self.best = current
        if self.wait >= self.patience:
            # We have stagnated
            self.wait = 0
            report = create_report(self.model, self.create_dataset())
            if len(self.inv_label) < self.max_invariants:
                idx = np.argmin(np.sum(report["inv_uni"], -1), 0)  # ()
                # print("Would have added:", report["input"][idx])
                # losses = tf.keras.losses.sparse_categorical_crossentropy(
                # report["label"], report["output"]
                # ).numpy()
                # idx = np.argmin(losses, 0)  # ()
                print("Adding new invariant with label:", report["label"][idx])
                for k in self.inv_inputs.keys():
                    self.inv_inputs[k] = np.concatenate(
                        [self.inv_inputs[k], report[k][None, idx]]
                    )
                self.inv_label = np.concatenate(
                    [self.inv_label, report["label"][None, idx]]
                )
                raise exceptions.NewInvariantException()


class EarlyStopAtConvergence(tf.keras.callbacks.Callback):
    """Early stop the training if the loss has reached a lower bound."""

    def __init__(self, converge_value: float = 0.01):
        super().__init__()
        self.converge_value = converge_value
        self.has_converged = False

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None):
        """Check for convergence."""
        logs = logs or dict()
        if logs["loss"] < self.converge_value:
            self.model.stop_training = True
            print(
                f"Model has converged to desired loss {logs['loss']} < {self.converge_value}."
            )
            mlflow.set_tag("result", "converged")


class TerminateOnNaN(tf.keras.callbacks.Callback):
    """Terminate training on NaN loss."""

    def on_batch_end(self, batch: int, logs: Dict[str, float] = None):
        """Check if loss is NaN."""
        logs = logs or dict()
        if tf.math.is_nan(logs["loss"]) or tf.math.is_inf(logs["loss"]):
            print(f"Batch {batch} has NaN or inf loss, terminating training.")
            self.model.stop_training = True
            mlflow.set_tag("result", "invalid")


class ParamScheduler(tf.keras.callbacks.Callback):
    """Schedule a change in model parameter."""

    def __init__(
        self,
        layer_params: List[Tuple[str, str]],
        scheduler: tf.keras.optimizers.schedules.LearningRateSchedule,
        min_value: float = None,
    ):
        super().__init__()
        self.layer_params = layer_params
        self.scheduler = scheduler
        self.min_value = min_value

    def get_parameter(self, layer_name: str, param_name: str) -> tf.Variable:
        """Return a reference to the scheduled parameter."""
        # Obtain reference to tf.Variable param
        layer: tf.keras.layers.Layer = self.model.get_layer(layer_name)
        param: tf.Variable = getattr(layer, param_name)
        return param

    def on_epoch_begin(self, epoch: int, logs: Dict[str, float] = None):
        """Apply schedule on parameter."""
        # Check if new value should be assigned
        for layer_name, param_name in self.layer_params:
            param = self.get_parameter(layer_name, param_name)
            new_value: tf.Tensor = self.scheduler(epoch)
            if self.min_value is None or new_value >= self.min_value:
                param.assign(new_value, read_value=False)

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None):
        """Append latest param value to logs."""
        # Append to log so other callbacks can see it / print it
        logs = logs or dict()
        for layer_name, param_name in self.layer_params:
            param = self.get_parameter(layer_name, param_name)
            logs[layer_name + "/" + param_name] = param.numpy()


class Evaluator(tf.keras.callbacks.Callback):
    """Evaluate model with multiple test datasets."""

    def __init__(
        self,
        datasets: Dict[str, tf.data.Dataset],
        dset_wrapper: Callable[[tf.data.Dataset], tf.data.Dataset] = None,
    ):
        super().__init__()
        self.datasets = datasets
        self.last_time = time.time()
        self.dset_wrapper = dset_wrapper or (lambda x: x)

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None):
        """Evaluate on every dataset and report metrics back."""
        logs = logs or dict()
        report: Dict[str, float] = {"epoch": float(epoch)}
        report.update({"train_" + k: v for k, v in logs.items()})
        for dname, dset in self.datasets.items():
            if not (dname.startswith("test") or dname.startswith("validation")):
                continue
            test_report: Dict[str, float] = self.model.evaluate(
                self.dset_wrapper(dset), verbose=0, return_dict=True
            )
            test_report = {dname + "_" + k: v for k, v in test_report.items()}
            report.update(test_report)
        # ---------------------------
        report["time"] = time.time() - self.last_time
        self.last_time = time.time()
        # Save and parint report
        print(
            " ".join([k + " " + "{:.3f}".format(v) for k, v in report.items()]),
        )
        # Add extra metrics back to logs so other callbacks can see it.
        logs.update(report)
        mlflow.log_metrics(report, step=epoch)


class ArtifactSaver(tf.keras.callbacks.Callback):
    """Save model artifacts after training is completed."""

    def __init__(
        self,
        datasets: Dict[str, tf.data.Dataset],
        artifact_dir: Path,
        dset_wrapper: Callable[[tf.data.Dataset], tf.data.Dataset] = None,
    ):
        super().__init__()
        self.datasets = datasets
        self.artifact_dir = artifact_dir
        self.dset_wrapper = dset_wrapper or (lambda x: x)

    def on_train_end(self, logs: Dict[str, float] = None):
        """Save generated model artifacts."""
        # Save model summary
        summary: List[str] = list()
        self.model.summary(print_fn=summary.append)
        art_dir = self.artifact_dir
        summary_path = art_dir / "model_summary.txt"
        with summary_path.open("w") as summary_file:
            summary_file.write("\n".join(summary))
        print("Saved model summary to", summary_path)
        # ---------------------------
        # Save model report artifacts
        for dname, dset in self.datasets.items():
            report = create_report(self.model, self.dset_wrapper(dset))
            fname = str(art_dir / (dname + "_report.npz"))
            np.savez_compressed(fname, **report)
            print("Saving model report artifact:", fname)
        # ---------------------------
        # Save artifacts to mlflow
        mlflow.log_artifacts(str(art_dir))
        # mlflow.keras.log_model(
        #     self.model, "mlflow_model", custom_objects=models.custom_layers
        # )
        # ---------------------------
        # Clean up
        shutil.rmtree(str(art_dir))
        try:
            art_dir.parent.rmdir()  # Delete only if empty
        except OSError:
            pass  # we will keep the directory
