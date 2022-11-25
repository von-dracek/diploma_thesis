import datetime
import enum

FIRST_VALID_DATE = datetime.datetime(2000, 1, 1, 1, 1)
LAST_VALID_DATE = datetime.datetime(2019, 12, 31, 23, 59)
DOWNLOAD_DATA = True
RETURNS_FILE = "returns"

MIN_NUMBER_LEAVES_IN_SCENARIO_TREE = 100
MAX_NUMBER_LEAVES_IN_SCENARIO_TREE = 700

TICKERS = ['ACGL', 'AFL', 'AIG', 'AJG', 'ALL', 'AON', 'AXP', 'BAC', 'BEN', 'BK', 'BLK', 'BRO', 'C', 'CB', 'CINF', 'CMA', 'COF', 'FDS', 'FITB', 'GL', 'GS', 'HBAN', 'HIG', 'IVZ', 'JPM', 'KEY', 'L', 'LNC', 'MCO', 'MMC', 'MS', 'MTB', 'NTRS', 'PGR', 'PNC', 'RE', 'RF', 'RJF', 'SCHW', 'SIVB', 'SPGI', 'STT', 'TFC', 'TROW', 'TRV', 'USB', 'WFC', 'WRB', 'ZION']
#randomly samples train and test tickers

train_tickers = ['ALL', 'BK', 'ACGL', 'FDS', 'ZION', 'TROW', 'STT', 'RJF', 'SPGI', 'AON', 'LNC', 'USB', 'AXP', 'AIG', 'AJG', 'BLK', 'RE', 'SIVB', 'NTRS', 'CB', 'WRB', 'BRO', 'L', 'IVZ', 'BAC', 'GS', 'WFC', 'MTB', 'MMC', 'BEN']
test_tickers = list(set(TICKERS).difference(set(train_tickers)))
assert len(set(train_tickers).intersection(set(test_tickers)))==0

class FILE(enum.Enum):
    RETURNS_FILE = "returns"
    DATA_PRICES_CLOSE_FILE = "Downloaded_Adjusted_Close_prices"

ASSET_SET_1 = train_tickers[:10]
ASSET_SET_2 = train_tickers[10:20]
ASSET_SET_3 = train_tickers[20:30]