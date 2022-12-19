from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.stats import kurtosis, skew

from src.configuration import FIRST_VALID_DATE, LAST_VALID_DATE
from src.data_downloading import date_add, date_intv


def get_TARMOM_and_R(dataset: pd.DataFrame) -> Tuple[Any, np.array]:
    corr_matrix = dataset.corr(method="pearson").to_numpy()
    assert np.all(
        np.linalg.eigvals(corr_matrix) > -1e-10
    ), "Correlation matrix R is not positive definite!"

    _dataset = dataset.to_numpy()
    mean = np.mean(_dataset, axis=0)
    variance = np.var(_dataset, axis=0)
    # skewness and kurtosis must be denormalised
    skewness = skew(_dataset, axis=0) * variance ** (3 / 2)
    kurt = kurtosis(_dataset, axis=0, fisher=False) * variance**2

    var_manual = np.sum(np.power(_dataset - mean, 2), axis=0) / _dataset.shape[0]
    skew_manual = np.sum(np.power(_dataset - mean, 3), axis=0) / _dataset.shape[0]
    kurt_manual = np.sum(np.power(_dataset - mean, 4), axis=0) / _dataset.shape[0]

    moments = np.array((mean, var_manual, skew_manual, kurt_manual))
    assert np.all(abs(var_manual - variance) < 1e-8)
    assert np.all(abs(skew_manual - skewness) < 1e-8)
    assert np.all(abs(kurt_manual - kurt) < 1e-8)

    return moments, corr_matrix


def data_to_returns_iid(data: pd.DataFrame, branching: List[int]):
    """Convert weekly data to returns according to specified branching.
    The time period is constant, only the number of stages changes - returns
    need to be calculated based on the given number of stages."""
    #the multiplication below is here so that we obtain a nontrivial number of return observations
    n_stages = len(branching) * 4
    first_date_in_dataset = data.index[0]
    last_date_in_dataset = data.index[-1]
    interval = date_intv(first_date_in_dataset, last_date_in_dataset, n_stages)
    timestamps = [first_date_in_dataset] + list(date_add(first_date_in_dataset, last_date_in_dataset, interval))

    nearest_dates = []
    for date in timestamps:
        nearest_date = _nearest_date(data.index, date)
        nearest_dates.append(nearest_date)
    data = data[data.index.isin(nearest_dates)]
    returns = data.pct_change() + 1
    #returns are calculated in such a way so that they can be multiplied together to get the final return
    returns = returns.dropna()
    assert len(returns) == len(branching) * 4
    return returns


def _nearest_date(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))