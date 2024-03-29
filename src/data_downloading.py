"""
This script can be used to download the necessary data. The used data are however located
in data/Downloaded_Adjusted_Close_prices_latest.parquet.
"""
import datetime
import pickle
from random import randint, sample, seed
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta

from src.configuration import (
    FILE,
    FIRST_VALID_DATE,
    LAST_VALID_DATE,
    TICKERS,
    test_tickers,
    train_tickers,
)


def download_data(
    start: datetime.datetime,
    end: datetime.datetime,
    tickers: Tuple[str, ...],
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        prepost=False,
        threads=True,
    )

    data = data.stack(level=0)
    data = data.drop([col for col in data.columns if col != "Close"], axis=1)
    data = data.unstack(level=1)
    data = data.sort_index()

    save_data(data=data, filename=FILE.DATA_PRICES_CLOSE_FILE.value)

    return data


def save_data(data: pd.DataFrame, filename: str) -> None:
    # with open(f"./data/{filename}_{datetime.datetime.now()}.pckl", "wb") as f:
    #     pickle.dump(data, f)
    with open(f"./data/{filename}_latest.parquet", "wb") as f:
        data.to_parquet(f)


def load_data(filename: str) -> pd.DataFrame:
    with open(f"./data/{filename}_latest.parquet", "rb") as f:
        return pd.read_parquet(f)


# deprecated function -- used before when we used pickle for storing the data
# but it is not clean to put .pckl files in attachment of thesis
def load_data_pickle(filename: str) -> pd.DataFrame:
    with open(f"./data/{filename}_latest.pckl", "rb") as f:
        return pickle.load(f)


def date_range(
    start: datetime.datetime, end: datetime.datetime, intv: int
) -> Iterator[str]:
    """intv - how many stages are there in the program (number of periods to create)"""
    # start = datetime.datetime.strptime(start,"%Y%m%d")
    # end = datetime.datetime.strptime(end,"%Y%m%d")
    diff = (end - start) / intv
    for i in range(intv):
        yield start + diff * i
    yield end


def date_add(
    start: datetime.datetime, end: datetime.datetime, diff: relativedelta
) -> Iterator[str]:
    """intv - how many stages are there in the program (number of periods to create)"""
    # start = datetime.datetime.strptime(start,"%Y%m%d")
    # end = datetime.datetime.strptime(end,"%Y%m%d")
    curr = start
    while curr < end:
        yield curr + diff
        curr += diff


def date_intv(
    start: datetime.datetime, end: datetime.datetime, intv: int
) -> datetime.timedelta:
    return (end - start) / intv


class DataGetter:
    def __init__(self):
        self.loaded_data = load_data(FILE.DATA_PRICES_CLOSE_FILE.value)
        self.first_half_of_data = self.loaded_data[: len(self.loaded_data) // 2]
        self.second_half_of_data = self.loaded_data[len(self.loaded_data) // 2 :]

    def randomly_sample_data(
        self,
        random_generator: np.random.default_rng,
        train_or_test: str,
        defined_tickers: List[str] = None,
        train_or_test_time: str = "train",
    ):
        """defined_tickers - list of tickers to choose from (not choosing randomly)"""
        tickers = (
            train_tickers
            if train_or_test == "train"
            else (test_tickers if train_or_test == "test" else None)
        )
        data = (
            self.first_half_of_data
            if train_or_test_time == "train"
            else (self.second_half_of_data if train_or_test_time == "test" else None)
        )
        n_stocks = random_generator.integers(7, 11)
        assert isinstance(tickers, List)
        chosen_tickers = list(random_generator.choice(tickers, n_stocks, replace=False))
        if defined_tickers is not None:
            chosen_tickers = defined_tickers
            print(f"WARNING!!!!! Fixing ticker set to {defined_tickers}")
        assert isinstance(chosen_tickers, List)
        chosen_tickers = [("Close", x) for x in chosen_tickers]
        return data[chosen_tickers]


if __name__ == "__main__":
    """Download data"""
    download_data(FIRST_VALID_DATE, LAST_VALID_DATE, TICKERS)
