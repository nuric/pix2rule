"""Utility functions for analysis."""
from typing import Dict, List, Tuple, Any
import pprint
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow


def delete_debug_experiments(mlclient: mlflow.tracking.MlflowClient):
    """Delete experiments recorded for debugging."""
    debug_exps: List[Tuple[int, str]] = [
        (exp.experiment_id, exp.name)
        for exp in mlclient.list_experiments()
        if exp.name.startswith("202")
    ]
    print("Experiments to delete:")
    pprint.pprint(debug_exps)
    user_input = input("Confirm? [y, N]")
    if user_input in ("y", "yes"):
        for exp in debug_exps:
            mlclient.delete_experiment(exp[0])
        print(f"Deleted {len(debug_exps)} experiments.")


# Collect all experiment data
def convert_maybe_to_number(input_str: str):
    """Maybe convert a string to a number."""
    for func in [int, float, str]:
        try:
            return func(input_str)
        except ValueError:
            continue


def collect_experiment_data(
    experiment_name: str, mlclient: mlflow.tracking.MlflowClient
) -> pd.DataFrame:
    """Collect all experiment run data into dataframe."""
    exp = mlclient.get_experiment_by_name(experiment_name)
    all_runs = mlclient.search_runs(exp.experiment_id)
    # Process data for each run
    run_pds = list()
    for run in all_runs:
        # run.data contains .metrics and .params both are dictionaries
        run_dict = dict()
        for metric in run.data.metrics.keys():
            # The following returns a list of Metric objects
            # which have .key .step .value and .timestamp
            mhist: List[mlflow.entities.Metric] = mlclient.get_metric_history(
                run.info.run_id, metric
            )
            mhist = sorted(mhist, key=lambda x: int(x.step))
            # Collect values into the dict
            run_dict[metric] = [h.value for h in mhist]
        # Append static params
        # run_dict['step'] = range(len(run_dict[next(iter(run.data.metrics.keys()))]))
        run_dict["run_id"] = run.info.run_id
        run_dict.update(
            {k: convert_maybe_to_number(v) for k, v in run.data.params.items()}
        )
        run_pds.append(pd.DataFrame(data=run_dict))
    return pd.concat(run_pds)


def plot_batch(
    data: Dict[str, np.ndarray],
    plot_func,
    size: int = 8,
    cols: int = 4,
    figsize=(12.8, 9.6),
):
    """Plot the data in subplots."""
    # data e.g. {'image': (B, ...), 'task_id': (B, ...)}
    plt.figure(figsize=figsize)
    size = size or data[next(iter(data.keys()))].shape[0]
    size = min(size, data[next(iter(data.keys()))].shape[0])
    rows = np.ceil(size / cols).astype(int)
    for i in range(size):
        plt.subplot(rows, cols, i + 1)
        plot_func({k: v[i] for k, v in data.items()})
    plt.show()


def load_artifact(
    run_id: str, fpath: str, mlclient: mlflow.tracking.MlflowClient
) -> Any:
    """Load run artifact."""
    # If mlflow is local, the following just returns a file path
    local_path: str = mlclient.download_artifacts(run_id, fpath)
    # ---------------------------
    # Load the file accordingly
    if local_path.endswith(".npz"):
        artifact = np.load(local_path)
    elif local_path.endswith(".json"):
        with open(local_path) as json_artifact:
            artifact = json.load(json_artifact)
    else:
        # Assume it is text file
        with open(local_path) as artifact_file:
            artifact = artifact_file.read()
    # ---------------------------
    return artifact
