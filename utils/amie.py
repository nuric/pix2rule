"""Utility functions for handling AMIE rule mining."""
from typing import Dict, List, Tuple
import subprocess
import tempfile
import itertools

import numpy as np

###############################################
# Using the default schema relations
# Assuming 9 as type relation
# Unexpected exception: Unrecognized option: --help
# usage: AMIE [OPTIONS] <TSV FILES>
#  -auta                             Avoid unbound type atoms, e.g., type(x,
#                                    y), i.e., bind always 'y' to a type
#  -bexr <body-excluded-relations>   Do not use these relations as atoms in
#                                    the body of rules. Example:
#                                    <livesIn>,<bornIn>
#  -bias <e-name>                    Syntatic/semantic bias:
#                                    oneVar|default|lazy|lazit|[Path to a
#                                    subclass of
#                                    amie.mining.assistant.MiningAssistant]D
#                                    efault: default (defines support and
#                                    confidence in terms of 2 head variables
#                                    given an order, cf -vo)
#  -btr <body-target-relations>      Allow only these relations in the body.
#                                    Provide a list of relation names
#                                    separated by commas (incompatible with
#                                    body-excluded-relations). Example:
#                                    <livesIn>,<bornIn>
#  -caos                             If a single variable bias is used
#                                    (oneVar), force to count support always
#                                    on the subject position.
#  -const                            Enable rules with constants. Default:
#                                    false
#  -d <delimiter>                    Separator in input files (default: TAB)
#  -datalog                          Print rules using the datalog notation
#                                    Default: false
#  -deml                             Do not exploit max length for speedup
#                                    (requested by the reviewers of AMIE+).
#                                    False by default.
#  -dpr                              Disable perfect rules.
#  -dqrw                             Disable query rewriting and caching.
#  -ef <extraFile>                   An additional text file whose
#                                    interpretation depends on the selected
#                                    mining assistant (bias)
#  -fconst                           Enforce constants in all atoms.
#                                    Default: false
#  -full                             It enables all enhancements: lossless
#                                    heuristics and confidence approximation
#                                    and upper bounds It overrides any other
#                                    configuration that is incompatible.
#  -hexr <head-excluded-relations>   Do not use these relations as atoms in
#                                    the head of rules (incompatible with
#                                    head-target-relations). Example:
#                                    <livesIn>,<bornIn>
#  -htr <head-target-relations>      Mine only rules with these relations in
#                                    the head. Provide a list of relation
#                                    names separated by commas (incompatible
#                                    with head-excluded-relations). Example:
#                                    <livesIn>,<bornIn>
#  -maxad <max-depth>                Maximum number of atoms in the
#                                    antecedent and succedent of rules.
#                                    Default: 3
#  -minc <min-std-confidence>        Minimum standard confidence threshold.
#                                    This value is not used for pruning,
#                                    only for filtering of the results.
#                                    Default: 0.0
#  -minhc <min-head-coverage>        Minimum head coverage. Default: 0.01
#  -minis <min-initial-support>      Minimum size of the relations to be
#                                    considered as head relations. Default:
#                                    100 (facts or entities depending on the
#                                    bias)
#  -minpca <min-pca-confidence>      Minimum PCA confidence threshold. This
#                                    value is not used for pruning, only for
#                                    filtering of the results. Default: 0.0
#  -mins <min-support>               Minimum absolute support. Default: 100
#                                    positive examples
#  -mlg                              Parse labels language as new facts
#  -nc <n-threads>                   Preferred number of cores. Round down
#                                    to the actual number of cores in the
#                                    system if a higher value is provided.
#  -noHeuristics                     Disable functionality heuristic, should
#                                    be used with the -full option
#  -noKbExistsDetection              Prevent the KB to detect existential
#                                    variable on-the-fly and to optimize the
#                                    query
#  -noKbRewrite                      Prevent the KB to rewrite query when
#                                    counting pairs
#  -oout                             If enabled, it activates only the
#                                    output enhacements, that is, the
#                                    confidence approximation and upper
#                                    bounds.  It overrides any other
#                                    configuration that is incompatible.
#  -optimai                          Prune instantiated rules that decrease
#                                    too much the support of their parent
#                                    rule (ratio 0.2)
#  -optimcb                          Enable the calculation of confidence
#                                    upper bounds to prune rules.
#  -optimfh                          Enable functionality heuristic to
#                                    identify potential low confident rules
#                                    for pruning.
#  -ostd                             Do not calculate standard confidence
#  -oute                             Print the rules at the end and not
#                                    while they are discovered. Default:
#                                    false
#  -pm <pruning-metric>              Metric used for pruning of intermediate
#                                    queries: support|headcoverage. Default:
#                                    headcoverage
#  -rl <recursivity-limit>           Recursivity limit
#  -verbose                          Maximal verbosity
#  -vo <variableOrder>               Define the order of the variable in
#                                    counting query among: app, fun
#                                    (default), ifun


def run_amie(
    knowledge_base: List[Tuple[str, str, str]], timeout: int = 3600, fpath: str = None
) -> str:
    """Run amie on given knowledge base."""
    # ---------------------------
    # Construct the tsv file string
    str_kb = "\n".join(["\t".join(t) for t in knowledge_base])
    # Optionally write the program file.
    if fpath:
        with open(fpath, "w") as fout:
            fout.write(str_kb)
    # ---------------------------
    # Run AMIE
    with tempfile.NamedTemporaryFile("w", suffix=".tsv") as temp_f:
        temp_f.write(str_kb)
        res = subprocess.run(
            [
                "java",
                "-jar",
                "/homes/nuric/bin/amie-milestone-intKB.jar",
                "-bexr",
                "target",
                "-datalog",
                "-full",
                "-htr",
                "target",
                "-maxad",
                "100",
                "-minhc",
                "1.0",
                temp_f.name,
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=timeout,
        )
    # ---------------------------
    return res.stdout


def generate_knowledge_base(
    dataset: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
) -> List[Tuple[str, str, str]]:
    """Convert batched interpretations into an iterable list of strings."""
    # dataset({'nullary': (B, P0), 'unary': (B, O, P1), 'binary': (B, O, O-1, P2)}, {'label': (B,)})
    # ---------------------------
    input_dict, output_dict = dataset
    batch_size = input_dict["nullary"].shape[0]  # B
    num_objects = input_dict["unary"].shape[1]  # O
    obj_bidxs = np.stack(np.nonzero(1 - np.eye(num_objects))).T  # (O*(O-1), 2)
    num_nullary = input_dict["nullary"].shape[-1]
    num_unary = input_dict["unary"].shape[-1]
    num_binary = input_dict["binary"].shape[-1]
    # ---------------------------
    knowledge_base: List[Tuple[str, str, str]] = list()
    for bidx in range(batch_size):
        nullary = input_dict["nullary"][bidx]  # (P0,)
        unary = input_dict["unary"][bidx]  # (O, P1)
        binary = input_dict["binary"][bidx]  # (O, O-1, P2)
        # ---------------------------
        eg_str = f"Example{bidx}"  # In AMIE constants start with capital
        # Add the label
        if output_dict["label"][bidx] == 1:
            knowledge_base.append((eg_str, "target", eg_str))
        # ---------------------------
        # Add objects
        knowledge_base.extend(
            [(eg_str, "has", f"Obj{i+bidx*num_objects}") for i in range(num_objects)]
        )
        # ---------------------------
        # Add nullary facts.
        knowledge_base.extend(
            [
                (eg_str, f"nullary{nullary[i] == 1}{i}", eg_str)
                for i in range(num_nullary)
            ]
        )
        # ---------------------------
        # Add unary facts.
        knowledge_base.extend(
            [
                (eg_str, f"unary{unary[i,j] == 1}{j}", f"Obj{i+bidx*num_objects}")
                for i, j in itertools.product(range(num_objects), range(num_unary))
            ]
        )
        # ---------------------------
        # Add binary facts.
        knowledge_base.extend(
            [
                (
                    f"Obj{i+bidx*num_objects}",
                    f"binary{binary[i, j - (j >= i), k] == 1}{k}",
                    f"Obj{j+bidx*num_objects}",
                )
                for (i, j), k in itertools.product(obj_bidxs, range(num_binary))
            ]
        )
        # ---------------------------
    return knowledge_base
