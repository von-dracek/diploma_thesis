import datetime
import pickle
from typing import Iterator, Tuple

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta

from src.configuration import FILE


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
