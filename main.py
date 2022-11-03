"""DOCSTRING"""

import random
from typing import List
import logging

import numpy as np

from gbm_generator import get_gbm_scenarios_from_TARMOM_and_R, get_gbm_scenarios_from_TARMOM_and_R_recursive
from src.configuration import (
    DOWNLOAD_DATA,
    FILE,
    FIRST_VALID_DATE,
    LAST_VALID_DATE,
    TICKERS,
)
from src.data_downloading import download_data, load_data
from src.data_preprocessing import data_to_returns_iid, get_TARMOM_and_R
from src.log import configure
from src.mean_cvar import calculate_mean_cvar_over_leaves
from src.tree_generation import create_empty_tree, fill_empty_tree_with_scenario_data

np.random.seed(1337)
random.seed(1337)
configure()


def get_cvar_value(branching: List[int]) -> float:
    """Start here."""

    # Downloading or loading data, based on DOWNLOAD_DATA flag defined in configuration.py
    if DOWNLOAD_DATA:
        data = download_data(start=FIRST_VALID_DATE, end=LAST_VALID_DATE, tickers=TICKERS)
    else:
        data = load_data(FILE.DATA_PRICES_CLOSE_FILE.value)

    logging.info(f"Generating cvar value for branching {branching}")
    iid_returns = data_to_returns_iid(data, branching)

    TARMOM, R = get_TARMOM_and_R(iid_returns)
    # get_gbm_scenarios_from_TARMOM_and_R_recursive(TARMOM, R, branching)
    # create scenario tree
    tree_root = create_empty_tree(branching)
    root = fill_empty_tree_with_scenario_data(TARMOM, R, tree_root, branching)
    return calculate_mean_cvar_over_leaves(root)


if __name__ == "__main__":
    branching = [3] * 5
    print(get_cvar_value(branching))
