import datetime
import enum

FIRST_VALID_DATE = datetime.datetime(2000, 1, 1, 1, 1)
LAST_VALID_DATE = datetime.datetime(2020, 12, 31, 23, 59)
DOWNLOAD_DATA = False
RETURNS_FILE = "returns"
TICKERS = tuple(sorted(["DLTR", "INTC", "MSFT", "AMZN", "SBUX", "QCOM", "DLTR", "MTCH"]))
N_STOCKS = len(TICKERS)
CVAR_ALPHA = 0.95
MAX_NUMBER_LEAVES_IN_SCENARIO_TREE = 1000



class FILE(enum.Enum):
    RETURNS_FILE = "returns"
    DATA_PRICES_CLOSE_FILE = "Downloaded_Adjusted_Close_prices"
