from typing import Dict, List, Union

import numpy as np
import pandas as pd

# TODO: maybe multiply moments by number of weeks in the given period (atleast mean)
from src.configuration import FIRST_VALID_DATE, LAST_VALID_DATE
from src.data_downloading import date_range


def get_TARMOM_and_R(dataset: pd.DataFrame) -> Dict[str, np.array]:
    corr_matrix = dataset.corr(method="pearson").to_numpy()
    assert np.all(
        np.linalg.eigvals(corr_matrix) > 0
    ), "Correlation matrix R is not positive definite!"

    _dataset = dataset.to_numpy()
    mean = np.mean(_dataset, axis=0)
    moments = np.array(
        (
            mean,
            np.sum(np.power(_dataset - mean, 2), axis=0) / _dataset.shape[0],
            np.sum(np.power(_dataset - mean, 3), axis=0) / _dataset.shape[0],
            np.sum(np.power(_dataset - mean, 4), axis=0) / _dataset.shape[0],
        )
    )
    return {"moments": moments, "R": corr_matrix}


def preprocess_data(
    returns: pd.DataFrame, branching: List[int]
) -> Dict[str, Dict[str, Union[pd.DataFrame, np.array]]]:
    n_stages = len(branching)
    equidistant_timestamps = list(date_range(FIRST_VALID_DATE, LAST_VALID_DATE, n_stages))
    available_data_up_to_timestamps = {
        timestamp: returns[:timestamp] for timestamp in equidistant_timestamps
    }

    # this dictionary holds data from previous timestamp up to the current timestamp
    # where the dictionary key corresponds to current timestamp
    data_obtained_between_timestamps = {}
    previous_timestamp = None
    for timestamp, df in available_data_up_to_timestamps.items():
        if previous_timestamp is None:
            previous_timestamp = FIRST_VALID_DATE
        data_obtained_between_timestamps[timestamp] = {"df": df[previous_timestamp:]}
        if not data_obtained_between_timestamps[timestamp]["df"].empty:
            data_obtained_between_timestamps[timestamp].update(
                get_TARMOM_and_R(data_obtained_between_timestamps[timestamp]["df"])
            )
        else:
            data_obtained_between_timestamps.pop(timestamp)
        previous_timestamp = timestamp

    return data_obtained_between_timestamps
