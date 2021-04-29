"""Experiment configuration generator."""
from typing import Any, Dict, List
import datetime
import json
import pprint
import random
import sys
from pathlib import Path

import mlflow
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
data_dir = Path(C["data_dir"])
configlib.save_config(str(data_dir / "defaults.json"))
print(f"There are {len(C.keys())} many configuration parameters.")
# ---------------------------

all_experiments: List[Dict[str, Any]] = list()
current_dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# ---------------------------
# Relsgame dataset with different training sizes
relsgame_exp = {
    "tracking_uri": "http://muli.doc.ic.ac.uk:8888",
    "experiment_name": "relsgame-" + current_dt,
    "max_steps": 30000,
    "eval_every": 200,  # divide max_steps by this to get epochs in keras
    "learning_rate": 0.001,
    "train_type": "deep",
    "dataset_name": "relsgame",
    "relsgame_train_size": [100, 1000, 5000],
    "relsgame_validation_size": 1000,
    "relsgame_test_size": 1000,
    "relsgame_batch_size": 64,
    "relsgame_output_type": "label",
    "run_count": list(range(5)),
    "relsgame_with_augmentation": True,
    "relsgame_noise_stddev": 0.01,
    "relsgame_rng_seed": 42,
}
# Setup fact sizes for a fairer comparison between different models.
# We want the hidden size to be the same.
NUM_UNARY_FEATS = 8
NUM_BINARY_FEATS = 16
objects_and_variables = {
    "same": (2, 2),
    "between": (3, 3),
    "occurs": (4, 2),
    "xoccurs": (4, 4),
    "all": (4, 4),
}
relsgame_datasets: List[Dict[str, Any]] = list()
for dname, (num_objects, num_variables) in objects_and_variables.items():
    facts_size = (
        num_objects * NUM_UNARY_FEATS
        + num_objects * (num_objects - 1) * NUM_BINARY_FEATS
    )
    relsgame_datasets.append(
        {
            "relsgame_task_nickname": dname,
            "relsgame_tasks": [dname] if dname != "all" else [],
            "dnf_image_classifier_object_sel_num_select": num_objects,
            "dnf_image_classifier_inference_num_total_variables": num_variables,
            "mlp_image_classifier_hidden_sizes": [facts_size, facts_size],
            "predinet_relations": NUM_BINARY_FEATS,
            "predinet_heads": facts_size // NUM_BINARY_FEATS,
        }
    )
# ---------------------------
# The following is the base dnf image classifier, it uses one iteration of
# inference with no hidden learnt predicates
base_dnf_image_classifier: Dict[str, Any] = {
    "nickname": "dnf_image_classifier",
    "model_name": "dnf_image_classifier",
    "dnf_image_classifier_image_layer_name": "RelationsGameImageInput",
    "dnf_image_classifier_image_hidden_size": 32,
    "dnf_image_classifier_image_activation": "relu",
    "dnf_image_classifier_image_with_position": True,
    "dnf_image_classifier_object_sel_layer_name": "RelaxedObjectSelection",
    "dnf_image_classifier_object_sel_initial_temperature": 0.5,
    "dnf_image_classifier_object_feat_layer_name": "LinearObjectFeatures",
    "dnf_image_classifier_object_feat_unary_size": NUM_UNARY_FEATS,
    "dnf_image_classifier_object_feat_binary_size": NUM_BINARY_FEATS,
    "dnf_image_classifier_object_feat_activation": "tanh",
    "dnf_image_classifier_hidden_arities": [],
    "dnf_image_classifier_hidden_layer_name": "WeightedDNF",
    "dnf_image_classifier_hidden_num_total_variables": 2,
    "dnf_image_classifier_hidden_num_conjuncts": 4,
    "dnf_image_classifier_hidden_recursive": False,
    "dnf_image_classifier_inference_layer_name": "WeightedDNF",
    "dnf_image_classifier_inference_arities": [],
    "dnf_image_classifier_inference_num_conjuncts": 8,
    "dnf_image_classifier_inference_recursive": False,
    "dnf_image_classifier_iterations": 1,
}
# The following model uses one hidden dnf layer by specifying the arities of
# the hidden predicates
dnf_image_classifier_hidden = base_dnf_image_classifier.copy()
# You can read the following as, 2 nullary, 4 unary and 8 binary predicates
hidden_arities = [0] * 2 + [1] * 4 + [2] * 8
dnf_image_classifier_hidden.update(
    {
        "nickname": "dnf_image_classifier_hidden",
        "dnf_image_classifier_hidden_arities": hidden_arities,
    }
)
# Finally the recursive model, which iterates twice with hidden predicates
dnf_image_classifier_recursive = base_dnf_image_classifier.copy()
hidden_arities = [0] * 1 + [1] * 2 + [2] * 4
dnf_image_classifier_recursive.update(
    {
        "nickname": "dnf_image_classifier_recursive",
        "dnf_image_classifier_inference_arities": hidden_arities,
        "dnf_image_classifier_inference_num_conjuncts": 2,
        "dnf_image_classifier_inference_recursive": True,
        "dnf_image_classifier_iterations": 2,
    }
)
dnf_image_classifier_models = [
    base_dnf_image_classifier,
    dnf_image_classifier_hidden,
    dnf_image_classifier_recursive,
]
# ---------------------------
# Setup dnf image classifier models that also perform reconstruction
recon_dnf_image_classifier_models: List[Dict[str, Any]] = list()
for mdict in dnf_image_classifier_models:
    mcopy = mdict.copy()
    mcopy.update(
        {
            "nickname": "recon_" + mdict["nickname"],
            "relsgame_output_type": "label_and_image",
        }
    )
    recon_dnf_image_classifier_models.append(mcopy)
# ---------------------------
# The MLP baseline models, 1 and 2 hidden layers
double_mlp = {
    "nickname": "mlp2",
    "model_name": "mlp_image_classifier",
    "mlp_image_classifier_image_layer_name": "RelationsGameImageInput",
    "mlp_image_classifier_image_hidden_size": 32,
    "mlp_image_classifier_image_activation": "relu",
    "mlp_image_classifier_image_with_position": True,
    "mlp_image_classifier_hidden_activations": ["relu", "relu"],
}
relsgame_predinet_model: Dict[str, Any] = {
    "nickname": "predinet",
    "model_name": "predinet",
    "predinet_image_layer_name": "RelationsGameImageInput",
    "predinet_image_hidden_size": 32,
    "predinet_image_activation": "relu",
    "predinet_image_with_position": True,
    "predinet_key_size": 32,
    "predinet_output_hidden_size": 64,
}
# ---
relsgame_exp["experiment_name"] = "relsgame-dnf-" + current_dt
relsgame_dnf_exps = hp.chain(
    hp.generate(relsgame_exp), relsgame_datasets, dnf_image_classifier_models
)
all_experiments.extend(relsgame_dnf_exps)
# ---
relsgame_exp["experiment_name"] = "relsgame-recon_dnf-" + current_dt
relsgame_dnf_exps = hp.chain(
    hp.generate(relsgame_exp), relsgame_datasets, recon_dnf_image_classifier_models
)
all_experiments.extend(relsgame_dnf_exps)
# ---
relsgame_exp["experiment_name"] = "relsgame-predinet-" + current_dt
relsgame_predinet_exps = hp.chain(
    hp.generate(relsgame_exp), relsgame_datasets, [relsgame_predinet_model]
)
all_experiments.extend(relsgame_predinet_exps)
# ---------------------------
# ILP experiments
gendnf_exp = {
    "tracking_uri": "http://muli.doc.ic.ac.uk:8888",
    "experiment_name": "gendnf-deep-" + current_dt,
    "max_steps": 10000,
    "eval_every": 200,  # divide max_steps by this to get epochs in keras
    "learning_rate": 0.001,
    "dataset_name": "gendnf",
    "gendnf_target_arity": 0,  # we are learning a single propositional rule
    "gendnf_gen_size": 10000,
    "gendnf_train_size": 2000,
    "gendnf_validation_size": 1000,
    "gendnf_test_size": 1000,
    "gendnf_batch_size": 128,
    # EuroMillions winning draw 25 December 2020
    "gendnf_rng_seed": [16, 21, 27, 30, 32, 3, 5],
}
gendnf_datasets = [
    {
        "gendnf_difficulty": "easy",
        "gendnf_num_objects": 3,
        "gendnf_num_nullary": 2,
        "gendnf_num_unary": 2,
        "gendnf_num_binary": 2,
        "gendnf_num_variables": 2,
        "gendnf_num_conjuncts": 3,
    },
    {
        "gendnf_difficulty": "medium",
        "gendnf_num_objects": 4,
        "gendnf_num_nullary": 4,
        "gendnf_num_unary": 5,
        "gendnf_num_binary": 6,
        "gendnf_num_variables": 3,
        "gendnf_num_conjuncts": 4,
    },
    {
        "gendnf_difficulty": "hard",
        "gendnf_num_objects": 4,
        "gendnf_num_nullary": 6,
        "gendnf_num_unary": 7,
        "gendnf_num_binary": 8,
        "gendnf_num_variables": 3,
        "gendnf_num_conjuncts": 5,
    },
]
gendnf_deep_models = hp.generate(
    {
        "train_type": "deep",
        "model_name": "dnf_rule_learner",
        "dnf_rule_learner_inference_layer_name": "WeightedDNF",
        "gendnf_input_noise_probability": [0.0, 0.15, 0.30],
        "run_count": list(range(5)),
    }
)
gendnf_deep_exps = hp.chain(
    hp.generate(gendnf_exp), gendnf_datasets, gendnf_deep_models
)
# all_experiments.extend(gendnf_deep_exps)
# ---
# Symbolic learners
gendnf_exp["experiment_name"] = "gendnf-ilp-" + current_dt
gendnf_ilp_models: List[Dict[str, Any]] = [
    {
        "train_type": "ilasp",
        "gendnf_return_numpy": True,
    },
    {
        "train_type": "fastlas",
        "gendnf_return_numpy": True,
    },
]
gendnf_ilp_exps = hp.chain(
    hp.generate(gendnf_exp), gendnf_datasets[:1], gendnf_ilp_models
)
# all_experiments.extend(gendnf_ilp_exps)
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
configs_path = data_dir / "experiments.json"
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
# Create mlflow experiment
mlflow.set_tracking_uri("http://localhost:8888")
for exp_name in set(exp["experiment_name"] for exp in all_experiments):
    exp_id = mlflow.create_experiment(exp_name)
    print("Created mlflow experiment:", exp_name, exp_id)
# ---------------------------
print("Done")
