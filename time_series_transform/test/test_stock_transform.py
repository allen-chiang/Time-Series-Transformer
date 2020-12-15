import pytest
import numpy as np
import pandas as pd
import pandas_ta as ta
from time_series_transform.io.pandas import (to_pandas,from_pandas)
from time_series_transform.stock_transform.base import (Stock,Portfolio)
from time_series_transform.stock_transform.stock_transfromer import Stock_Transformer
from time_series_transform.transform_core_api.base import (Time_Series_Data,Time_Series_Data_Collection)

@pytest.fixture('class')
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

@pytest.fixture('class')
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

class Test_Stock_Transform:

    def test_stock_dtype(self,dictList_stock,dictList_portfolio):
        data = dictList_stock
        stock = Stock(data,'Date')
        st = Stock_Transformer(data,'Date',None,'symbol')
        assert stock == st.time_series_data
        data = dictList_portfolio
        port = Portfolio(Stock(data,'Date'),'Date','symbol')
        st = Stock_Transformer(data,'Date','symbol')
        assert port == st.time_series_data

    def test_single_from_pandas(self,dictList_stock):
        df = pd.DataFrame(dictList_stock)
        stockTrans = Stock_Transformer.from_pandas(df,'Date',None)
        test = Stock_Transformer(dictList_stock,'Date',None)
        assert stockTrans == test

    def test_single_from_numpy(self,dictList_stock):
        df = pd.DataFrame(dictList_stock).values
        stockTrans = Stock_Transformer.from_numpy(df,0,None,1,2,3,4,5)
        test = Stock_Transformer(pd.DataFrame(df),0,None,1,2,3,4,5)
        assert stockTrans == test

    def test_single_make_technical_indicator(self):
        pass

    def test_collection_from_pandas(self,dictList_portfolio):
        df = pd.DataFrame(dictList_portfolio)
        st = Stock_Transformer.from_pandas(df,'Date','symbol')
        test = Stock_Transformer(dictList_portfolio,'Date','symbol')
        assert test == st

    def test_collection_from_numpy(self,dictList_portfolio):
        df = pd.DataFrame(dictList_portfolio).values
        stockTrans = Stock_Transformer.from_numpy(df,0,6,1,2,3,4,5)
        test = Stock_Transformer(pd.DataFrame(df),0,6,1,2,3,4,5)
        assert stockTrans == test

    def test_collection_make_technical_indicator(self):
        pass