"""Script to submit condor jobs for experiments."""
from typing import Dict, Set
import json
import os
import pprint
import subprocess
import sys
from pathlib import Path

import mlflow
from mlflow.entities import RunStatus

# We run bash as executable because Condor does not follow the virtual
# environment symbolic link executable properly i.e., it thinks it is running
# system Python rather than the virtual env one.
TEMPLATE = r"""notification = Error
notify_user = nuric@imperial.ac.uk
executable = /usr/bin/bash
error  = data/outs/$(Cluster)-$(Process).err
output = data/outs/$(Cluster)-$(Process).out
log    = data/outs/$(Cluster)-$(Process).log
requirements = Target.OpSysAndVer == "Ubuntu20" && regexp("(edge|point|ray|sprite|texel|vertex)\d+.doc.ic.ac.uk", Target.Machine)
rank = (-Target.TotalLoadAvg) + regexp("vertex\d+.doc.ic.ac.uk", Target.Machine)*4
arguments = "-c 'data/venv/bin/python3 train.py --config_json $(Item)'"
environment = "PATH=/homes/nuric/bin:/usr/bin:/usr/local/bin LD_LIBRARY_PATH=/homes/nuric/bin/lib"
queue from (
{configs}
)"""


def parse_condor_job_ads() -> Dict[str, str]:
    """Parse on going Condor job ads."""
    jobad_fpath = os.environ.get("_CONDOR_JOB_AD")
    # ---------------------------
    # Check if we are running inside Condor
    if not jobad_fpath:
        return dict()
    # ---------------------------
    # Parse the file
    job_ads: Dict[str, str] = dict()
    with open(jobad_fpath) as fin:
        # line example: Owner = "user"\n
        for line in fin:
            line = line.strip()
            key, value = line.split(" = ", 1)
            job_ads[key] = value.strip('"')
    # ---------------------------
    return job_ads


def main():
    """Check and submit condor jobs from experiments json."""
    # ---------------------------
    # Load unfinished experiments
    mlflow.set_tracking_uri("http://localhost:8888")
    with open("data/experiments.json") as experiments_file:
        experiments = json.load(experiments_file)
    experiment_names = set(exp["experiment_name"] for exp in experiments.values())
    experimen_ids = [
        mlflow.get_experiment_by_name(exp_name).experiment_id
        for exp_name in experiment_names
    ]
    all_config_jsons = set("data/experiments.json." + k for k in experiments.keys())
    # ---------------------------
    # Search for completed runs
    all_runs = mlflow.search_runs(experiment_ids=experimen_ids)
    run_status: Dict[str, Set[str]] = dict()
    # RunStatus enum ['RUNNING', 'SCHEDULED', 'FINISHED', 'FAILED', 'KILLED']
    for stat in RunStatus.all_status():
        stat_str = RunStatus.to_string(stat)
        filtered_runs = all_runs.loc[all_runs["status"] == stat_str]
        run_status[stat_str] = set()
        if not filtered_runs.empty:
            run_status[stat_str] = set(filtered_runs["params.config_json"].values)
    # ---------------------------
    # Here is the situation
    pprint.pprint({k: len(v) for k, v in run_status.items()})
    # ---------------------------
    # Find which runs still need to be run
    # RunStatus enum ['RUNNING', 'SCHEDULED', 'FINISHED', 'FAILED', 'KILLED']
    to_ignore = set.union(
        *[run_status[s] for s in ["RUNNING", "SCHEDULED", "FINISHED"]]
    )
    still_to_run = all_config_jsons - to_ignore
    if not still_to_run:
        print("Nothing to run...")
        sys.exit(0)
    # ---------------------------
    # Ask for permission
    print("---------------------------")
    print("Still to complete:", len(still_to_run))
    if input("Do you wish to proceed? (y,N): ") not in ["y", "yes"]:
        print("Aborting experiment run...")
        sys.exit(1)
    # ---------------------------
    # Create output folder, Condor complains if it does not exist
    Path("data/outs").mkdir(exist_ok=True, parents=True)
    # ---------------------------
    # Generate condor script and run
    condor_cmd = TEMPLATE.format(configs="\n".join(still_to_run))
    with open("temp_condor.txt", "w") as fout:
        fout.write(condor_cmd)
    subprocess.run(["condor_submit", "temp_condor.txt"], check=True)
    # ---------------------------
    # Clean up
    os.remove("temp_condor.txt")
    print("Done.")


if __name__ == "__main__":
    main()
