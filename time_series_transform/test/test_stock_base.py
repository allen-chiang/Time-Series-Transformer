import pytest
import numpy as np
import pandas as pd
import pandas_ta as ta
from time_series_transform.io.pandas import (to_pandas,from_pandas)
from time_series_transform.stock_transform.base import (Stock,Portfolio)
from time_series_transform.transform_core_api.base import (Time_Series_Data,Time_Series_Data_Collection)

@pytest.fixture(scope = 'class')
def dictList_stock():
    return {
        'Date': ['2020-01-01', '2020-01-02'],
        'Close': [1, 2],
        'Open': [1, 2],
        'Low': [1, 2],
        'High': [1, 2],
        'Volume': [1, 2],
        'symbol':['AT','AT']
    }

@pytest.fixture(scope = 'class')
def dictList_portfolio():
    return {
        'Date': ['2020-01-01', '2020-01-02','2020-01-01', '2020-01-02'],
        'Close': [1,2,1,2],
        'Open': [1,2,1,2],
        'Low': [1,2,1,2],
        'High': [1,2,1,2],
        'Volume': [1,2,1,2],
        'symbol':['AT','AT','GOOGL','GOOGL']
    }

class Test_Stock:
    
    def test_from_time_series_tensor(self,dictList_stock):
        data = dictList_stock
        tsd = Time_Series_Data(data,'Date')
        test = Stock(data,'Date')
        stock = Stock.from_time_series_data(tsd,None)
        assert test == stock


    def test_get_technical_indicator(self,dictList_stock):
        data = dictList_stock
        df = pd.DataFrame(data)
        stock = Stock(data,'Date')
        strategy = ta.Strategy(
            name = 'sma',
            ta = [
                {'kind':'sma','length':2}
            ]
        )
        df.ta.strategy(strategy)
        stock = stock.get_technical_indicator(strategy)
        test = to_pandas(stock,False,False,'ignore')
        test.columns = test.columns.str.lower()
        df.columns = df.columns.str.lower()
        pd.testing.assert_frame_equal(df,test,False)


class Test_Portfolio:

    def test_from_time_series_collection(self,dictList_portfolio):
        data = dictList_portfolio
        tsd = Time_Series_Data(data,'Date')
        tsc = Time_Series_Data_Collection(tsd,'Date','symbol')
        port = Portfolio.from_time_series_collection(tsc)
        test = Portfolio(tsd,'Date','symbol')
        assert port == test

    def test_get_technical_indicator(self,dictList_portfolio):
        data = dictList_portfolio
        df = pd.DataFrame(data)
        tsd = Time_Series_Data(data,'Date')
        tsc = Time_Series_Data_Collection(tsd,'Date','symbol')
        port = Portfolio.from_time_series_collection(tsc)
        strategy = ta.Strategy(
            name = 'sma',
            ta = [
                {'kind':'sma','length':2}
            ]
        )
        port = port.get_technical_indicator(strategy)
        test = to_pandas(port,False,False,'ignore')
        test.columns = test.columns.str.lower()
        for i in port:
            tmp = df[df.symbol == i]
            tmp.ta.strategy(strategy)
            tmp.columns = tmp.columns.str.lower()
            tmp_test = test[test.symbol == i]
            tmp_test = tmp_test[tmp.columns]
            pd.testing.assert_frame_equal(tmp_test.reset_index(drop=True),tmp.reset_index(drop=True),False)