import datetime
import enum

FIRST_VALID_DATE = datetime.datetime(2000, 1, 1, 1, 1)
LAST_VALID_DATE = datetime.datetime(2020, 12, 31, 23, 59)
DOWNLOAD_DATA = True
RETURNS_FILE = "returns"
TICKERS = tuple(
    sorted(["VRTX", "ASML", "AMD", "SBUX", "QCOM", "DLTR", "MTCH"])
)
# TICKERS = tuple(["AMD", "SBUX"])
N_STOCKS = len(TICKERS)
CVAR_ALPHA = 0.95

class FILE(enum.Enum):
    RETURNS_FILE = "returns"
    DATA_PRICES_CLOSE_FILE = "Downloaded_Adjusted_Close_prices"
