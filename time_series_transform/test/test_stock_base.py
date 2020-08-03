import pytest
from time_series_transform.stock_transform.base import *
from time_series_transform.stock_transform.stock_extractor import *

@pytest.mark.parametrize("source", ["yahoo"])
@pytest.mark.parametrize("stockSymbol", ["aapl", "0050.TW", "GC=F"])
class Test_stock_extractor:
    @pytest.mark.parametrize("periods", ["1y", "max", "1d"])
    def test_extractor_get_stock_period(self, stockSymbol, periods,source):
        se = Stock_Extractor(stockSymbol,source)
        data = se.get_stock_period(periods)
        assert isinstance(data, Stock)

    @pytest.mark.parametrize("dates", [["2019-01-01", "2020-07-01"], ["2015-01-01", "2020-01-01"],["2020-01-01", "2020-01-01"]])
    def test_extractor_get_stock_date(self,stockSymbol, dates, source):
        se = Stock_Extractor(stockSymbol,source)
        data = se.get_stock_date(dates[0], dates[1])
        assert isinstance(data, Stock)

@pytest.mark.parametrize("x", [5])
class Test_stock:
    @pytest.mark.parametrize("y", [5])
    def test_cls(self, x,y):
        assert x+y == 10

    class test_t:
        @pytest.mark.parametrize("y", [9])
        def test_clas(self, x,y):
            assert x+y == 14


