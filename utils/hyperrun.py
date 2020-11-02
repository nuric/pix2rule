"""Script to run experiments."""
from typing import Dict, Any, Iterator
import itertools
from . import hashing


def generate_configs(
    hconfig: Dict[str, Any], with_hash=False
) -> Iterator[Dict[str, Any]]:
    """Generate every config from lists in a dictionary."""
    lists = {k: v for k, v in hconfig.items() if isinstance(v, list)}
    notlists = {k: v for k, v in hconfig.items() if not isinstance(v, list)}
    # Compute cross product between list entry values
    for instance in itertools.product(*lists.values()):
        config_dict = dict(zip(lists.keys(), instance))
        config_dict.update(notlists)
        if with_hash:
            config_dict["hash"] = hashing.dict_hash(config_dict)
        yield config_dict
