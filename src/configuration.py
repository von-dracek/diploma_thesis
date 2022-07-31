import datetime
import enum

FIRST_VALID_DATE = datetime.datetime(2011, 1, 1, 1, 1)
LAST_VALID_DATE = datetime.datetime(2020, 12, 31, 23, 59)
DOWNLOAD_DATA = False
RETURNS_FILE = "returns"
TICKERS = tuple(
    sorted(["VRTX", "ASML", "AMD", "SBUX", "NFLX", "TSLA", "QCOM", "DLTR", "AMGN", "MTCH"])
)
N_STOCKS = len(TICKERS)


class FILE(enum.Enum):
    RETURNS_FILE = "returns"
    DATA_PRICES_CLOSE_FILE = "Downloaded_Adjusted_Close_prices"
