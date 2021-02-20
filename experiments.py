"""Experiment configuration generator."""
from typing import Any, Dict, List
import argparse
import json
import pprint
import random
import sys
from pathlib import Path

import utils.hashing
import utils.hyperrun as hp

# ---------------------------
# Configuration parameters
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--data_dir", default="data", help="Folder for experiment data.")
ARGS = parser.parse_args()
# ---------------------------

all_experiments: List[Dict[str, Any]] = list()
# ---------------------------
# Relsgame dataset with different training sizes
relsgame_exp = {
    "experiment_name": "relsgame_full",
    "max_steps": 30000,
    "eval_every": 200,  # divide max_steps by this to get epochs in keras
    "learning_rate": [0.001],
    "dataset_name": "relsgame",
    "relsgame_tasks": [
        "same",
        "between",
        "occurs",
        "xoccurs",
        "colour_and_or_shape",
        "",
    ],
    "relsgame_train_size": [10000, 1000, 100, 10],
    "relsgame_validation_size": 1000,
    "relsgame_test_size": 1000,
    "relsgame_batch_size": 64,
}
relsgame_models = [
    {
        "model_name": "dnf_image_classifier",
        "relsgame_one_hot_labels": True,
    }
]
relsgame_exps = hp.chain(hp.generate(relsgame_exp), relsgame_models)
all_experiments.extend(relsgame_exps)
# ---------------------------
# Ask user if they are on the right path (?)
print("---------------------------")
print("Here is a random sample:")
pprint.pprint(random.choice(all_experiments))
print("---------------------------")
print("Total number of runs:", len(all_experiments))
print("---------------------------")
if input("Do you wish to proceed? (y,N): ") not in ["y", "yes"]:
    print("Aborting experiment generation...")
    sys.exit(1)
# ---------------------------
# Generate configuration index
configs_index = {utils.hashing.dict_hash(exp): exp for exp in all_experiments}
# ---------------------------
# Pre-experiment data generation if any
# ---------------------------
# Write configuration file
configs_path = Path(ARGS.data_dir) / "experiments.json"
print("Writing configurations to", configs_path)
with configs_path.open("w") as configs_file:
    json.dump(configs_index, configs_file, indent=4)
