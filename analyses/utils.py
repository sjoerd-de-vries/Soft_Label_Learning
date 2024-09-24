import sys

from soft_label_learning.experiments.experiment_settings import (
    ens_methods,
    method_keys,
    multi_label_methods,
    single_label_methods,
)


def deep_getsizeof(o, ids):
    """Find the memory footprint of a Python object
    This is a recursive function that drills down into the contents of objects and
    calculates their memory footprint.
    """
    d = sys.getsizeof(o)
    if id(o) in ids:
        return 0

    ids.add(id(o))

    if isinstance(o, dict):
        return d + sum(
            deep_getsizeof(k, ids) + deep_getsizeof(v, ids) for k, v in o.items()
        )

    if isinstance(o, (list, tuple, set, frozenset)):
        return d + sum(deep_getsizeof(i, ids) for i in o)

    return d


def total_size(o):
    """Calculate total size of object accounting for objects it references"""
    return deep_getsizeof(o, set())


def get_linestyle_dict():
    linestyle_dict = {}

    for method in method_keys:
        if method in single_label_methods:
            linestyle = "-"
            if method in ens_methods:
                linestyle = "-."
        elif method in multi_label_methods:
            linestyle = (0, (5, 10))
            if method in ens_methods:
                linestyle = "--"
        else:
            raise ValueError(f"Method {method} not found")

        linestyle_dict[method] = linestyle

    return linestyle_dict
