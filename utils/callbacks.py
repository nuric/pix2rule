"""Custom training utility callbacks."""
from typing import List, Dict
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import mlflow
import mlflow.keras

import models
from configlib import config as C
from reportlib import create_report


class EarlyStopAtConvergence(tf.keras.callbacks.Callback):
    """Early stop the training if the loss has reached a lower bound."""

    def __init__(self, converge_value: float = 0.01):
        super(EarlyStopAtConvergence, self).__init__()
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


class Evaluator(tf.keras.callbacks.Callback):
    """Evaluate model with multiple test datasets."""

    def __init__(self, datasets: Dict[str, tf.data.Dataset]):
        super(Evaluator, self).__init__()
        self.datasets = datasets

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None):
        """Evaluate on every dataset and report metrics back."""
        logs = logs or dict()
        report = {"train_" + k: v for k, v in logs.items()}
        for dname, dset in self.datasets.items():
            if not dname.startswith("test"):
                continue
            test_report: Dict[str, float] = self.model.evaluate(dset, return_dict=True)
            test_report = {dname + "_" + k: v for k, v in test_report.items()}
            report.update(test_report)
        # ---------------------------
        # Save and parint report
        print(" ".join([k + " " + "{:.3f}".format(v) for k, v in report.items()]))
        mlflow.log_metrics(report, step=epoch)


class ArtifactSaver(tf.keras.callbacks.Callback):
    """Save model artifacts after training is completed."""

    def __init__(self, datasets: Dict[str, tf.data.Dataset]):
        super(ArtifactSaver, self).__init__()
        self.datasets = datasets

    def on_train_end(self, logs: Dict[str, float] = None):
        """Save generated model artifacts."""
        # Save model summary
        summary: List[str] = list()
        self.model.summary(print_fn=summary.append)
        art_dir = Path(C["artifact_dir"])
        summary_path = art_dir / "model_summary.txt"
        with summary_path.open("w") as summary_file:
            summary_file.write("\n".join(summary))
        print("Saved model summary to", summary_path)
        # ---------------------------
        # Save model report artifacts
        for dname, dset in self.datasets.items():
            report = create_report(self.model, dset)
            fname = str(art_dir / (dname + "_report.npz"))
            np.savez_compressed(fname, **report)
            print("Saving model report artifact:", fname)
        # ---------------------------
        # Save artifacts to mlflow
        mlflow.log_artifacts(str(art_dir))
        mlflow.keras.log_model(
            self.model, "mlflow_model", custom_objects=models.custom_layers
        )
        # ---------------------------
        # Clean up
        shutil.rmtree(str(art_dir))
        try:
            art_dir.parent.rmdir()  # Delete only if empty
        except OSError:
            pass  # we will keep the directory
