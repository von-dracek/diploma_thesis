"""DOCSTRING"""

from src.configuration import (
    DOWNLOAD_DATA,
    FILE,
    FIRST_VALID_DATE,
    LAST_VALID_DATE,
    TICKERS,
)
from src.data_downloading import download_data, load_data
from src.data_preprocessing import preprocess_data, data_to_returns_iid, get_TARMOM_and_R
from src.log import configure
from src.mean_cvar import calculate_mean_cvar_over_leaves
from src.tree_generation import create_empty_tree, fill_empty_tree_with_scenario_data
import random
import numpy as np
np.random.seed(1337)
random.seed(1337)
configure()


def main() -> None:
    """Start here."""

    # Downloading or loading data, based on DOWNLOAD_DATA flag defined in configuration.py
    if DOWNLOAD_DATA:
        data = download_data(start=FIRST_VALID_DATE, end=LAST_VALID_DATE, tickers=TICKERS)
    else:
        data = load_data(FILE.DATA_PRICES_CLOSE_FILE.value)

    branching = [7, 8, 5, 3]
    iid_returns = data_to_returns_iid(data, branching)

    # preprocessed data contains data obtained from the
    # previous timestamp up until the current timestamp
    # preprocessed_data = preprocess_data(returns, branching)
    # preprocessed_data_iid = preprocess_data_iid_stages(iid_returns, branching)
    TARMOM, R = get_TARMOM_and_R(iid_returns)
    # create scenario tree
    tree_root = create_empty_tree(branching)
    root = fill_empty_tree_with_scenario_data(TARMOM, R, tree_root, branching)
    calculate_mean_cvar_over_leaves(root)
    # fill_empty_tree_with_scenario_data(preprocessed_data_iid, tree_root)
    # fill_empty_tree_with_scenario_data(preprocessed_data, tree_root)
    print("")


if __name__ == "__main__":
    main()
