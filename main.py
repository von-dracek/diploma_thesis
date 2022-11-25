"""DOCSTRING"""

import logging
import random
from typing import List

import numpy as np

from src.configuration import (
    FILE,
)
from src.data_downloading import DataGetter
from src.data_preprocessing import data_to_returns_iid, get_TARMOM_and_R
from src.log import configure
from src.mean_cvar import calculate_mean_cvar_over_leaves
from src.tree_generation import create_empty_tree, fill_empty_tree_with_scenario_data_moment_matching

np.random.seed(1337)
random.seed(1337)
configure()

datagetter = DataGetter()

def get_necessary_data(random_generator, train_or_test:str = "train", defined_tickers: List[str]=None):
    data = datagetter.randomly_sample_data(random_generator, train_or_test, defined_tickers)
    return data

def get_cvar_value(gams_workspace, branching: List[int], data, alpha: float = 0.95, train_or_test:str = "train") -> float:
    iid_returns = data_to_returns_iid(data, branching)

    TARMOM, R = get_TARMOM_and_R(iid_returns)
    # create scenario tree
    logging.info(f"Generating tree for branching {branching}")
    tree_root = create_empty_tree(branching)
    root = fill_empty_tree_with_scenario_data_moment_matching(TARMOM, R, tree_root, branching, gams_workspace)
    return calculate_mean_cvar_over_leaves(root, alpha, gams_workspace)


if __name__ == "__main__":
    """Start here."""
    branching = [2,3]
    data = get_necessary_data()
    print(get_cvar_value(branching, data))
