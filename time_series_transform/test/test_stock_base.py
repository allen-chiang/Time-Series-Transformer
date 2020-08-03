import pytest
from time_series_transform.stock_transform.base import *
from time_series_transform.stock_transform.stock_extractor import *

@pytest.mark.parametrize("stockSymbol, period, date", [
    ("aapl", "1y", ["2019-01-01", "2020-07-01"]),
    ("0050.TW", "max", ["2015-01-01", "2020-01-01"]),
    ("GC=F", "1d", ["2020-01-01", "2020-01-01"])
])

def test_extractor_get_stock_period(stockSymbol, period, date):
    for symbol in stockSymbol:
        se = Stock_Extractor(symbol,'yahoo')
        for p in period:
            data = se.get_stock_period(p)
            assert isinstance(data, Stock)

def test_extractor_get_stock_date(stockSymbol, date, period):
    for symbol in stockSymbol:
        se = Stock_Extractor(symbol,'yahoo')
        for d in date:
            data = se.get_stock_date(d[0], d[1])
            assert isinstance(data, Stock)
            