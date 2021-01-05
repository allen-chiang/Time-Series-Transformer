import pytest
import numpy as np
import pandas as pd
import os
import pandas_ta as ta
import pyarrow as pa
from time_series_transform.io.pandas import (to_pandas,from_pandas)
from time_series_transform.stock_transform.base import (Stock,Portfolio)
from time_series_transform.stock_transform.stock_transfromer import Stock_Transformer
from time_series_transform.transform_core_api.base import (Time_Series_Data,Time_Series_Data_Collection)
from time_series_transform.transform_core_api.time_series_transformer import Time_Series_Transformer

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
        test = Stock_Transformer(pd.DataFrame(df),0,None,None,1,2,3,4,5)
        assert stockTrans == test

    def test_single_make_technical_indicator(self,dictList_stock):
        MyStrategy = ta.Strategy(
            name="DCSMA10",
            ta=[
               {"kind": "sma", "length": 1}
            ]
        )
        df = pd.DataFrame(dictList_stock)
        stockTrans = Stock_Transformer.from_pandas(df,'Date',None)
        stockTrans = stockTrans.get_technial_indicator(MyStrategy)
        df.ta.strategy(MyStrategy)
        test = stockTrans.to_pandas(False,False,'ignore')
        test.columns = test.columns.str.lower()
        df.columns = df.columns.str.lower()
        df = df[test.columns]
        pd.testing.assert_frame_equal(test,df,False)

    def test_collection_from_pandas(self,dictList_portfolio):
        df = pd.DataFrame(dictList_portfolio)
        st = Stock_Transformer.from_pandas(df,'Date','symbol')
        test = Stock_Transformer(dictList_portfolio,'Date','symbol')
        assert test == st

    def test_collection_from_numpy(self,dictList_portfolio):
        df = pd.DataFrame(dictList_portfolio).values
        stockTrans = Stock_Transformer.from_numpy(df,0,6,1,2,3,4,5)
        test = Stock_Transformer(pd.DataFrame(df),0,6,None,1,2,3,4,5)
        assert stockTrans == test

    def test_collection_make_technical_indicator(self,dictList_portfolio):
        MyStrategy = ta.Strategy(
            name="DCSMA10",
            ta=[
               {"kind": "sma", "length": 1}
            ]
        )
        df = pd.DataFrame(dictList_portfolio)
        stockTrans = Stock_Transformer.from_pandas(df,'Date',None)
        stockTrans = stockTrans.get_technial_indicator(MyStrategy)
        test = stockTrans.to_pandas(False,False,'ignore')
        test.columns = test.columns.str.lower()
        for i in test.symbol.unique():
            tmp_df = df[df.symbol == i]
            tmp_test = test[test.symbol==i]
            tmp_df.ta.strategy(MyStrategy)
            tmp_df.columns = tmp_df.columns.str.lower()
            tmp_df = tmp_df[tmp_test.columns]
            pd.testing.assert_frame_equal(tmp_test,tmp_df,False)


    def test_from_time_series_transform(self,dictList_stock,dictList_portfolio):
        data = pd.DataFrame(dictList_stock)
        tst = Time_Series_Transformer.from_pandas(data,'Date',None)
        stockTrans = Stock_Transformer.from_pandas(data,'Date',None)
        test = Stock_Transformer.from_time_series_transformer(tst)
        assert test == stockTrans
        data = pd.DataFrame(dictList_portfolio)
        tst = Time_Series_Transformer.from_pandas(data,'Date','symbol')
        stockTrans = Stock_Transformer.from_pandas(data,'Date','symbol')
        test = Stock_Transformer.from_time_series_transformer(tst)
        assert stockTrans == test


    def test_from_arrow_table(self,dictList_stock,dictList_portfolio):
        data = pd.DataFrame(dictList_stock)
        table = pa.Table.from_pandas(data)
        stockTrans = Stock_Transformer.from_pandas(data,'Date',None)
        test = Stock_Transformer.from_arrow_table(table,'Date',None)
        assert test == stockTrans
        data = pd.DataFrame(dictList_portfolio)
        table = pa.Table.from_pandas(data)
        stockTrans = Stock_Transformer.from_pandas(data,'Date','symbol')
        test = Stock_Transformer.from_arrow_table(table,'Date','symbol')
        assert stockTrans == test

    def test_from_parquet(self,dictList_stock,dictList_portfolio):
        data = pd.DataFrame(dictList_stock)
        data.to_parquet('./data.parquet')
        stockTrans = Stock_Transformer.from_pandas(data,'Date',None)
        test = Stock_Transformer.from_parquet('./data.parquet','Date',None)
        assert test == stockTrans
        data = pd.DataFrame(dictList_portfolio)
        data.to_parquet('./data.parquet')
        stockTrans = Stock_Transformer.from_pandas(data,'Date','symbol')
        test = Stock_Transformer.from_parquet('./data.parquet','Date','symbol')
        assert stockTrans == test
        os.remove('./data.parquet')

    def test_from_feather(self,dictList_stock,dictList_portfolio):
        data = pd.DataFrame(dictList_stock)
        data.to_feather('./data.feather')
        stockTrans = Stock_Transformer.from_pandas(data,'Date',None)
        test = Stock_Transformer.from_feather('./data.feather','Date',None)
        assert test == stockTrans
        data = pd.DataFrame(dictList_portfolio)
        data.to_feather('./data.feather')
        stockTrans = Stock_Transformer.from_pandas(data,'Date','symbol')
        test = Stock_Transformer.from_feather('./data.feather','Date','symbol')
        assert stockTrans == test
        os.remove('./data.feather')