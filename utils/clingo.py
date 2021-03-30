"""Thin wrapper around command line clingo."""
from typing import Dict, List
import re
import itertools
import subprocess
import os

import numpy as np
import tqdm


def clingo_check(
    interpretation: Dict[str, np.ndarray], rule: Dict[str, np.ndarray]
) -> np.ndarray:
    """Generates logic program files and runs them with clingo to check satisfiability."""
    # interpretation {'nullary': (B, P0), 'unary': (B, O, P1), 'binary': (B, O, O-1, P2)}
    # rule {'and_kernel': (H, IN), 'or_kernel': (H,), 'num_variables': int}
    # ---------------------------
    num_nullary = interpretation["nullary"].shape[-1]
    num_unary = interpretation["unary"].shape[-1]
    num_binary = interpretation["binary"].shape[-1]
    num_variables = rule["num_variables"]
    # ---------------------------
    var_bidxs = np.stack(np.nonzero(1 - np.eye(num_variables))).T  # (V*(V-1), 2)
    pred_names = (
        [f"nullary({i})" for i in range(num_nullary)]
        + [
            f"unary(V{i},{j})"
            for i, j in itertools.product(range(num_variables), range(num_unary))
        ]
        + [
            f"binary(V{i},V{j},{k})"  # There is this modulus business because we omit p(X,X)
            for (i, j), k in itertools.product(var_bidxs, range(num_binary))
        ]
    )  # This is a mess that produces "n0..n, p1..n(V1), p1...n(V2), etc"
    # Here is a full dump for our sanity:
    # ['nullary(0)', 'unary(V0,0)', 'unary(V0,1)', 'unary(V1,0)',
    # 'unary(V1,1)', 'binary(V0,V1,0)', 'binary(V0,V1,1)', 'binary(V1,V0,0)',
    # 'binary(V1,V0,1)']
    # ---------------------------
    # target rule must be satisfied, to get satisfiable output
    rule_lines: List[str] = [":- not t."]
    # ---------------------------
    # Construct the rule
    conjuncts: List[str] = list()
    for conjunct in rule["and_kernel"]:
        conjs = [
            n if i == 1 else "not " + n for n, i in zip(pred_names, conjunct) if i != 0
        ]
        # ['n0', '-p0(V1)', ...]
        # ---------------------------
        # Let's also add the uniqueness of variables assumption
        conj_vars = [
            v for c in conjs for v in re.findall(r"V\d", c)
        ]  # ['V1', 'V2', ...]
        conj_vars = list(set(conj_vars))
        for i, first_var in enumerate(conj_vars):
            conjs.append(f"obj({first_var})")
            for second_var in conj_vars[i + 1 :]:
                conjs.append(f"{first_var} != {second_var}")
        # ---------------------------
        cstr = ", ".join(conjs)
        conjuncts.append(cstr)
    # Target rule t is now defined according to which conjuncts are in the rule
    assert np.all(
        rule["or_kernel"] != -1
    ), "Clingo cannot work with negated conjunctions inside a disjunction."
    # or_kernel [0, 1, 0, 0, ...] x H
    disjuncts = [
        "t :- {}.".format(conjuncts[i])
        for i in range(len(conjuncts))
        if rule["or_kernel"][i] != 0
    ]
    rule_lines.extend(disjuncts)
    # ---------------------------
    # That's the rule done, now add the interpretation
    batch_size = interpretation["nullary"].shape[0]  # B
    num_objects = interpretation["unary"].shape[1]  # O
    rule_lines.append(f"obj(0..{num_objects-1}).")
    results: List[bool] = list()
    for bidx in tqdm.trange(batch_size):
        logic_program = rule_lines.copy()
        nullary = interpretation["nullary"][bidx]  # (P0,)
        unary = interpretation["unary"][bidx]  # (O, P1)
        binary = interpretation["binary"][bidx]  # (O, O-1, P2)
        # ---------------------------
        # Add nullary facts.
        logic_program.extend(
            [f"nullary({i})." if nullary[i] == 1 else "" for i in range(num_nullary)]
        )
        # ---------------------------
        # Add unary facts.
        logic_program.extend(
            [
                f"unary({i},{j})." if unary[i, j] == 1 else ""
                for i, j in itertools.product(range(num_objects), range(num_unary))
            ]
        )
        # ---------------------------
        # Add binary facts.
        obj_bidxs = np.stack(np.nonzero(1 - np.eye(num_objects))).T  # (O*(O-1), 2)
        logic_program.extend(
            [
                f"binary({i},{j},{k})." if binary[i, j - (j >= i), k] == 1 else ""
                for (i, j), k in itertools.product(obj_bidxs, range(num_binary))
            ]
        )
        # ---------------------------
        # Write the program file.
        with open("clingo_temp.lp", "w") as fout:
            fout.write("\n".join(logic_program))
        # ---------------------------
        # Ask clingo the result
        # Clingo for some reason returns exit code 10 on success, we won't have return check
        res = subprocess.run(
            ["clingo", "clingo_temp.lp"], capture_output=True, check=False, text=True
        )
        results.append("UNSATISFIABLE" not in res.stdout)
    # ---------------------------
    # Clean up
    os.remove("clingo_temp.lp")
    # ---------------------------
    return np.array(results)  # (B,)
