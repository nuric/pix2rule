"""Script to run experiments."""
from typing import Dict, Any, List
import itertools
from . import hashing


def generate(hconfig: Dict[str, Any], with_hash=False) -> List[Dict[str, Any]]:
    """Generate every config from lists in a dictionary."""
    lists = {k: v for k, v in hconfig.items() if isinstance(v, list)}
    notlists = {k: v for k, v in hconfig.items() if not isinstance(v, list)}
    # Compute cross product between list entry values
    configs: List[Dict[str, Any]] = list()
    for instance in itertools.product(*lists.values()):
        config_dict = dict(zip(lists.keys(), instance))
        config_dict.update(notlists)
        if with_hash:
            config_dict["hash"] = hashing.dict_hash(config_dict)
        configs.append(config_dict)
    return configs


def chain(*configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chain given iterable config dictionaries."""
    chained_configs: List[Dict[str, Any]] = list()
    for instance in itertools.product(*configs):
        merged_config: Dict[str, Any] = dict()
        # instance (dict_a, dict_b, ...)
        for instance_dict in instance:
            merged_config.update(instance_dict)
        chained_configs.append(merged_config)
    return chained_configs
