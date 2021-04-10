"""ILASP utility interface."""
from typing import Dict, Any, List, Tuple
import logging
import subprocess
import pprint

import tensorflow as tf

from . import clingo

logger = logging.getLogger(__name__)

gendnf_sizes = [
    {
        "gendnf_difficulty": "easy",
        "gendnf_num_objects": 4,
        "gendnf_num_nullary": 7,
        "gendnf_num_unary": 8,
        "gendnf_num_binary": 9,
        "gendnf_num_variables": 4,
        "gendnf_num_conjuncts": 2,
    },
    # {
    #     "gendnf_difficulty": "medium",
    #     "gendnf_num_objects": 4,
    #     "gendnf_num_nullary": 2,
    #     "gendnf_num_unary": 3,
    #     "gendnf_num_binary": 4,
    #     "gendnf_num_variables": 2,
    #     "gendnf_num_conjuncts": 3,
    # },
    # {
    #     "gendnf_difficulty": "hard",
    #     "gendnf_num_objects": 4,
    #     "gendnf_num_nullary": 4,
    #     "gendnf_num_unary": 5,
    #     "gendnf_num_binary": 6,
    #     "gendnf_num_variables": 4,
    #     "gendnf_num_conjuncts": 4,
    # },
]


def _dummy_task_description(data_config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a dummy task description to use for hypothesis space generation."""
    # data_config {
    # "gendnf_num_objects": 4,
    # "gendnf_num_nullary": 1,
    # "gendnf_num_unary": 1,
    # "gendnf_num_binary": 2,
    # "gendnf_num_variables": 2,
    # "gendnf_num_conjuncts": 2,
    task_description = {
        "inputs": {
            "nullary": {"shape": [data_config["gendnf_num_nullary"]]},
            "unary": {
                "shape": [
                    data_config["gendnf_num_objects"],
                    data_config["gendnf_num_unary"],
                ]
            },
            "binary": {"shape": [data_config["gendnf_num_binary"]]},
        },
        "metadata": {"num_variables": data_config["gendnf_num_variables"]},
    }
    return task_description


def generate_search_space(task_description: Dict[str, Any]) -> Tuple[List[str], int]:
    """Genereate the las file lines for the desired search space."""
    # Generate training file
    num_nullary = task_description["inputs"]["nullary"]["shape"][-1]  # num_nullary
    num_unary = task_description["inputs"]["unary"]["shape"][-1]  # num_unary
    num_binary = task_description["inputs"]["binary"]["shape"][-1]  # num_binary
    num_objects = task_description["inputs"]["unary"]["shape"][1]  # O
    num_variables = task_description["metadata"]["num_variables"]  # V
    # ---------------------------
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
    # ---------------------------
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
    # ---------------------------
    # Add constants
    for ctype, count in [
        ("obj", num_objects),
        ("nullary_type", num_nullary),
        ("unary_type", num_unary),
        ("binary_type", num_binary),
    ]:
        for i in range(count):
            lines.append(f"#constant({ctype}, {i}).")
    # ---------------------------
    # Add max penalty for ILASP
    max_size = (num_nullary + unary_size + binary_size) * 3
    lines.append(f"#max_penalty({max_size}).")
    # ---------------------------
    return lines, max_size


def generate_pos_examples(dset: tf.data.Dataset) -> List[str]:
    """Generate file lines for positive examples."""
    lines: List[str] = list()
    # ---------------------------
    examples = {**dset[0], **dset[1]}
    # {'nullary': (B, P0), 'unary': (B, O, P1), 'binary': (B, O, O-1, P2), 'label': (B,)}
    string_examples = clingo.tensor_interpretations_to_strings(
        examples
    )  # list of lists
    for i, (str_example, label) in enumerate(zip(string_examples, dset[1]["label"])):
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
    return lines


def run_ilasp(
    fpath: str, max_size: int, only_search_space: bool = False, timeout: int = 3600
) -> str:
    """Run ILASP on the given file, optionally just generate the search space."""
    # ---------------------------
    # Construct the ILASP call
    # We assume here that ILASP is available in $PATH
    ilasp_cmd = [
        "ILASP",
        "--version=4",
        "--no-constraints",
        "--no-aggregates",
        f"-ml={max_size}",
        f"--max-rule-length={max_size}",
        "--strict-types",
        fpath,
    ]
    if only_search_space:
        ilasp_cmd.append("-s")
    # ---------------------------
    # Call ILASP
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
    # ---------------------------
    return res.stdout


def get_search_space_size(task_description: Dict[str, Any]) -> int:
    """Generate and get the size of the search space for ILASP."""
    lines, max_size = generate_search_space(task_description)
    # ---------------------------
    # Write to temporary file
    with open("ilasp_temp.lp", "w") as fout:
        fout.write("\n".join(lines))
    # ---------------------------
    # Call ILASP to generate the search space
    logger.info("Asking ILASP for the search space.")
    res = run_ilasp("ilasp_temp.lp", max_size, only_search_space=True)
    res_lines = [l for l in res.split("\n") if l]  # Remove empty lines
    assert res_lines[0] == "1 ~ t.", f"ILASP search space started with {res_lines[0]}."
    # ---------------------------
    print(res)
    return len(res_lines)


if __name__ == "__main__":
    # Generate and print search spaces
    print("Generating sizes for configurations:", len(gendnf_sizes))
    print("---------------------------")
    for dconf in gendnf_sizes:
        pprint.pprint(dconf)
        print("Size:", get_search_space_size(_dummy_task_description(dconf)))
        print("---------------------------")
