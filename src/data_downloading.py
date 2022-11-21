import datetime
import pickle
from typing import Iterator, Tuple, List
from random import randint, sample, seed
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta

from src.configuration import FILE, train_tickers, test_tickers, FIRST_VALID_DATE, LAST_VALID_DATE, TICKERS


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
    with open(f"./data/{filename}_latest.pckl", "wb") as f:
        pickle.dump(data, f)


def load_data(filename: str) -> pd.DataFrame:
    with open(f"./data/{filename}_latest.pckl", "rb") as f:
        return pickle.load(f)


def date_range(start: datetime.datetime, end: datetime.datetime, intv: int) -> Iterator[str]:
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


def date_intv(start: datetime.datetime, end: datetime.datetime, intv: int) -> datetime.timedelta:
    return (end - start) / intv

class DataGetter:
    def __init__(self):
        self.loaded_data = load_data(FILE.DATA_PRICES_CLOSE_FILE.value)
        self.first_half_of_data = self.loaded_data[:len(self.loaded_data)//2]
        self.second_half_of_data = self.loaded_data[len(self.loaded_data)//2:]

    def randomly_sample_data(self, train_or_test: str):
        tickers = train_tickers if train_or_test == "train" else (test_tickers if train_or_test=="test" else None)
        data = self.first_half_of_data if train_or_test == "train" else (self.second_half_of_data if train_or_test=="test" else None)
        n_stocks = randint(4,10)
        assert isinstance(tickers, List)
        chosen_tickers = sample(tickers, n_stocks)
        assert isinstance(chosen_tickers, List)
        chosen_tickers = [("Close", x) for x in chosen_tickers]
        return data[chosen_tickers]


if __name__ == "__main__":
    """Start here."""
    download_data(FIRST_VALID_DATE, LAST_VALID_DATE, TICKERS)
