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
    n_stages = len(branching)
    interval = date_intv(LAST_VALID_DATE, LAST_VALID_DATE + relativedelta(years=10), n_stages)
    timestamps = list(date_add(FIRST_VALID_DATE, LAST_VALID_DATE, interval))

    nearest_dates = []
    for date in timestamps:
        nearest_date = _nearest_date(data.index, date)
        if nearest_date != data.index[-1]:
            # the last interval has to be cut off, since it happens that the last interval would be too short
            # we have investment horizon of 10 years and got data for the last 20 years. According to the specified branching,
            # we partition the 20years of data into intervals of length 10years/len(branching). This means that the
            # last interval is longer that the 20 year cutoff (if the branching does not divide
            # both 10 and 20 years exactly) - we need to drop the last interval
            nearest_dates.append(nearest_date)
    data = data[data.index.isin(nearest_dates)]
    returns = data.pct_change() + 1
    #returns are calculated in such a way so that they can be multiplied together to get the final return
    #todo: maybe consider log returns?
    # log_returns = np.log(data) - np.log(data.shift(1)) + 1
    return returns.dropna()


def _nearest_date(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


# def preprocess_data(
#     returns: pd.DataFrame, branching: List[int]
# ) -> Dict[str, Dict[str, Union[pd.DataFrame, np.array]]]:
#     n_stages = len(branching)
#     equidistant_timestamps = list(date_range(FIRST_VALID_DATE, LAST_VALID_DATE, n_stages))
#     available_data_up_to_timestamps = {
#         timestamp: returns[:timestamp] for timestamp in equidistant_timestamps
#     }
#
#     # this dictionary holds data from previous timestamp up to the current timestamp
#     # where the dictionary key corresponds to current timestamp
#     data_obtained_between_timestamps = {}
#     previous_timestamp = None
#     for timestamp, df in available_data_up_to_timestamps.items():
#         if previous_timestamp is None:
#             previous_timestamp = FIRST_VALID_DATE
#         data_obtained_between_timestamps[timestamp] = {"df": df[previous_timestamp:]}
#         if not data_obtained_between_timestamps[timestamp]["df"].empty:
#             data_obtained_between_timestamps[timestamp].update(
#                 get_TARMOM_and_R(data_obtained_between_timestamps[timestamp]["df"])
#             )
#         else:
#             data_obtained_between_timestamps.pop(timestamp)
#         previous_timestamp = timestamp
#
#     return data_obtained_between_timestamps
