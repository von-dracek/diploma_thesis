"""DOCSTRING"""

from src.configuration import (
    DOWNLOAD_DATA,
    FILE,
    FIRST_VALID_DATE,
    LAST_VALID_DATE,
    TICKERS,
)
from src.data_downloading import download_data, load_data
from src.data_preprocessing import preprocess_data
from src.log import configure
from src.tree_generation import create_empty_tree, fill_empty_tree_with_scenario_data

configure()


def main() -> None:
    """Start here."""

    # Downloading or loading data, based on DOWNLOAD_DATA flag defined in configuration.py
    if DOWNLOAD_DATA:
        _, returns = download_data(start=FIRST_VALID_DATE, end=LAST_VALID_DATE, tickers=TICKERS)
    else:
        returns = load_data(FILE.RETURNS_FILE.value)

    branching = [12, 8, 5, 3]

    # preprocessed data contains data obtained from the
    # previous timestamp up until the current timestamp
    preprocessed_data = preprocess_data(returns, branching)

    # create scenario tree
    tree_root = create_empty_tree(branching)
    fill_empty_tree_with_scenario_data(preprocessed_data, tree_root)
    print(preprocessed_data)
    print("")


if __name__ == "__main__":
    main()
