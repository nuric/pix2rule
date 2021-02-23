"""Experiment configuration generator."""
from typing import Any, Dict, List
import json
import pprint
import random
import sys
from pathlib import Path

import configlib

import utils.hashing
import utils.hyperrun as hp

# This will import all parameters as well
import train  # pylint: disable=unused-import
import datasets

# ---------------------------
# Parse configuration parameters
configlib.parse()
C = configlib.config
configlib.save_config(str(Path(C["data_dir"]) / "defaults.json"))
print(f"There are {len(C.keys())} many configuration parameters.")
# ---------------------------

all_experiments: List[Dict[str, Any]] = list()
# ---------------------------
# Relsgame dataset with different training sizes
relsgame_exp = {
    "experiment_name": "relsgame_1k_full",
    "max_steps": 30000,
    "eval_every": 200,  # divide max_steps by this to get epochs in keras
    "learning_rate": 0.001,
    "dataset_name": "relsgame",
    "relsgame_tasks": [
        ["same"],
        ["between"],
        ["occurs"],
        ["xoccurs"],
        ["colour_and_or_shape"],
        [],
    ],
    "relsgame_train_size": 1000,
    "relsgame_validation_size": 1000,
    "relsgame_test_size": 1000,
    "relsgame_batch_size": 64,
}
relsgame_models: List[Dict[str, Any]] = [
    {
        "model_name": "dnf_image_classifier",
        "dnf_image_layer_name": "RelationsGameImageInput",
        "dnf_image_hidden_size": 32,
        "dnf_image_activation": "relu",
        "dnf_image_noise_stddev": 0.0,
        "dnf_image_with_position": True,
        "dnf_object_sel_layer_name": "RelaxedObjectSelection",
        "dnf_object_sel_num_select": 4,
        "dnf_object_sel_initial_temperature": 0.5,
        "dnf_object_feat_layer_name": "LinearObjectFeatures",
        "dnf_object_feat_unary_size": 6,
        "dnf_object_feat_binary_size": 12,
        "dnf_hidden_predicates": [],
        "dnf_iterations": 2,
        "relsgame_one_hot_labels": True,
    },
    {
        "model_name": "predinet",
        "predinet_image_layer_name": "RelationsGameImageInput",
        "predinet_image_hidden_size": 32,
        "predinet_image_activation": "relu",
        "predinet_image_noise_stddev": 0.0,
        "predinet_image_with_position": True,
        "predinet_relations": 12,
        "predinet_heads": 8,
        "predinet_key_size": 32,
        "predinet_output_hidden_size": 64,
    },
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
configs_path = Path(C["data_dir"]) / "experiments.json"
print("Writing configurations to", configs_path)
with configs_path.open("w") as configs_file:
    json.dump(configs_index, configs_file, indent=4)
# ---------------------------
# Generate data for the experiments
data_paths: List[str] = list()
for exp in all_experiments:
    C.update(exp)
    data_paths.append(datasets.get_dataset().generate_data())
print("Generated data files:")
pprint.pprint(data_paths)
print(f"Total of {len(data_paths)}.")
# ---------------------------
print("Done")
