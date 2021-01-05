import os
import copy
import pytest
import numpy as np
import collections
import pandas as pd
import pandas_ta as ta
from time_series_transform.sklearn.transformer import (
    Base_Time_Series_Transformer,
    Lag_Transformer,
    Function_Transformer,
    Stock_Technical_Indicator_Transformer
)
from time_series_transform.transform_core_api.time_series_transformer import *

@pytest.fixture(scope = 'class')
def data_input_single():
    return {
        'train':{
            'time':[1,2,3],
            'data1':[1,2,3],
            'data2':[-2,-2,-2]
        },
        'test':{
            'time':[4,5],
            'data1':[4,5],
            'data2':[-2,-2]
        }
    }

@pytest.fixture(scope = 'class')
def data_input_collection():
    return {
        'train':{
            'time': [1,2,1,2,3],
            'data':[1,1,1,2,3],
            'category':[1,1,2,2,2]
        },
        'test':{
            'time': [3,4,4,5],
            'data':[1,1,4,5],
            'category':[1,1,2,2]
        }
    }


@pytest.fixture(scope = 'class')
def expected_lag_output_collection():
    return {
        'train_withTimeCategory':{
            'time': [1,2,1,2,3],
            'data':[np.nan,1,np.nan,1,2],
            'category':[1,1,2,2,2]
        },
        'test_withTimeCategory':{
            'time': [3,4,4,5],
            'data':[1,1,3,4],
            'category':[1,1,2,2]
        },
        'train_withCategory':{
            'data':[np.nan,1,np.nan,1,2],
            'category':[1,1,2,2,2],
        },
        'test_withCategory':{
            'data':[1,1,3,4],
            'category':[1,1,2,2],
        },
        'train_withoutTimeCategory':{
            'data':[np.nan,1,np.nan,1,2],
        },
        'test_withoutTimeCategory':{
            'data':[1,1,3,4],
        },
        'train_withTime':{
            'time': [1,2,1,2,3],
            'data':[np.nan,1,np.nan,1,2],
        },
        'test_withTime':{
            'time': [3,4,4,5],
            'data':[1,1,3,4],
        },
    }


@pytest.fixture(scope = 'class')
def expected_lag_output_single():
    return {
        'train_rmTime':{
            'data1':[np.nan,1,2],
            'data2':[np.nan,-2,-2]
        },
        'train_withTime':{
            'time':[1,2,3],
            'data1':[np.nan,1,2],
            'data2':[np.nan,-2,-2]
        }, 
        'test_rmTime':{
            'data1':[3,4],
            'data2':[-2,-2]
        },
        'test_withTime':{
            'time':[4,5],
            'data1':[3,4],
            'data2':[-2,-2]
        }
    }

@pytest.fixture(scope = 'class')
def expected_func_output_single():
    return {
        'train_rmTime':{
            'data1':[2,3,4],
            'data2':[-1,-1,-1]
        },
        'train_withTime':{
            'time':[1,2,3],
            'data1':[2,3,4],
            'data2':[-1,-1,-1]
        }, 
        'test_rmTime':{
            'data1':[5,6],
            'data2':[-1,-1]
        },
        'test_withTime':{
            'time':[4,5],
            'data1':[5,6],
            'data2':[-1,-1]
        }
    }

@pytest.fixture(scope = 'class')
def expected_func_output_collection():
    return {
        'train_withTimeCategory':{
            'time': [1,2,1,2,3],
            'data':[2,2,2,3,4],
            'category':[1,1,2,2,2]
        },
        'test_withTimeCategory':{
            'time': [3,4,4,5],
            'data':[2,2,5,6],
            'category':[1,1,2,2]
        },
        'train_withCategory':{
            'data':[2,2,2,3,4],
            'category':[1,1,2,2,2],
        },
        'test_withCategory':{
            'data':[2,2,5,6],
            'category':[1,1,2,2],
        },
        'train_withoutTimeCategory':{
            'data':[2,2,2,3,4],
        },
        'test_withoutTimeCategory':{
            'data':[2,2,5,6],
        },
        'train_withTime':{
            'time': [1,2,1,2,3],
            'data':[2,2,2,3,4],
        },
        'test_withTime':{
            'time': [3,4,4,5],
            'data':[2,2,5,6],
        },
    }
 
@pytest.fixture(scope = 'class')
def single_stock():
    return {
    'train':{
        'Date':[1,2,3,4,5,6,7,8,9,10],
        'Close':[1,2,3,4,5,6,7,8,9,10],
        'High':[1,2,3,4,5,6,7,8,9,10],
        'Volume':[1,2,3,4,5,6,7,8,9,10],
        'Open':[1,2,3,4,5,6,7,8,9,10],
        'Low':[1,2,3,4,5,6,7,8,9,10]
        },
    'test':{
        'Date':[11,12,13],
        'Close':[11,12,13],
        'High':[11,12,13],
        'Volume':[11,12,13],
        'Open':[11,12,13],
        'Low':[11,12,13]
        }
    }

@pytest.fixture(scope = 'class')
def collection_stock():
    return {
        'train':{
            'Date':[1,2,3,4,5,1,2,3,4,5],
            'Close':[1,2,3,4,5,1,2,3,4,5],
            'Open':[1,2,3,4,5,1,2,3,4,5],
            'High':[1,2,3,4,5,1,2,3,4,5],
            'Low':[1,2,3,4,5,1,2,3,4,5],
            'Volume':[1,2,3,4,5,1,2,3,4,5],
            'symbol':[1,1,1,1,1,2,2,2,2,2]
        },
        'test':{
            'Date':[6,7,6,7],
            'Close':[6,7,6,7],
            'Open':[6,7,6,7],
            'High':[6,7,6,7],
            'Low':[6,7,6,7],
            'Volume':[6,7,6,7],
            'symbol':[1,1,2,2]
        }
    }



class Test_sklearn_transformer:

    def test_base_input_single(self,data_input_single):
        df = pd.DataFrame(data_input_single['train'])
        numpyData = df.values
        transformer = Base_Time_Series_Transformer('time',None,'ignore')
        transformer = transformer.fit(df)
        print(transformer.get_time_series_index_cache())
        results,timeStamp,headers,category = transformer.transform(df)
        tst = Time_Series_Transformer.from_pandas(df,'time',None)
        assert tst == results
        assert transformer.get_time_series_index_cache() == df.time.tolist()
        transformer = Base_Time_Series_Transformer(0,None,'ignore')
        transformer = transformer.fit(numpyData)
        results,timeStamp,headers,category = transformer.transform(numpyData)
        tst = Time_Series_Transformer.from_numpy(numpyData,0,None)
        assert tst == results
        assert transformer.get_time_series_index_cache() == df.time.tolist()
        
    def test_base_input_collection(self,data_input_collection):
        df = pd.DataFrame(data_input_collection['train'])
        numpyData = df.values
        transformer = Base_Time_Series_Transformer('time','category','ignore')
        transformer = transformer.fit(df)
        print(transformer.get_time_series_index_cache())
        results,timeStamp,headers,category = transformer.transform(df)
        tst = Time_Series_Transformer.from_pandas(df,'time','category')
        assert tst == results
        assert transformer.get_time_series_index_cache() == df.time.tolist()
        transformer = Base_Time_Series_Transformer(0,2,'ignore')
        transformer = transformer.fit(numpyData)
        results,timeStamp,headers,category = transformer.transform(numpyData)
        tst = Time_Series_Transformer.from_numpy(numpyData,0,2)
        assert tst == results
        assert transformer.get_time_series_index_cache() == df.time.tolist()


    def test_lag_single(self,data_input_single,expected_lag_output_single):
        df_train = pd.DataFrame(data_input_single['train'])
        numpyData_train = df_train.values
        expectDict = {}
        for i in expected_lag_output_single:
            expectDict[i]=pd.DataFrame(expected_lag_output_single[i]).values
        df_test = pd.DataFrame(data_input_single['test'])
        numpyData_test = df_test.values
        transformer = Lag_Transformer(1,'time')
        transformer = transformer.fit(df_train)
        train = transformer.transform(df_train)
        test = transformer.transform(df_test)
        np.testing.assert_equal(train,expectDict['train_rmTime'])
        np.testing.assert_equal(test,expectDict['test_rmTime'])
        transformer = Lag_Transformer(1,'time',remove_time=False)
        transformer = transformer.fit(df_train)
        train = transformer.transform(df_train)
        test = transformer.transform(df_test)
        np.testing.assert_equal(train,expectDict['train_withTime'])
        np.testing.assert_equal(test,expectDict['test_withTime'])


    def test_lag_collection(self,data_input_collection,expected_lag_output_collection):
        df_train = pd.DataFrame(data_input_collection['train'])
        df_test = pd.DataFrame(data_input_collection['test'])
        expectDict = {}
        for i in expected_lag_output_collection:
            expectDict[i]=pd.DataFrame(expected_lag_output_collection[i]).values
        transformer = Lag_Transformer(1,'time','category')
        transformer = transformer.fit(df_train)
        train = transformer.transform(df_train)
        test = transformer.transform(df_test)
        np.testing.assert_equal(train,expectDict['train_withoutTimeCategory'])
        np.testing.assert_equal(test,expectDict['test_withoutTimeCategory'])


    def test_lag_collection_withCategory(self,data_input_collection,expected_lag_output_collection):
        df_train = pd.DataFrame(data_input_collection['train'])
        df_test = pd.DataFrame(data_input_collection['test'])
        expectDict = {}
        for i in expected_lag_output_collection:
            expectDict[i]=pd.DataFrame(expected_lag_output_collection[i]).values
        transformer = Lag_Transformer(1,'time','category',remove_category=False)
        transformer = transformer.fit(df_train)
        train = transformer.transform(df_train)
        test = transformer.transform(df_test)
        np.testing.assert_equal(train,expectDict['train_withCategory'])
        np.testing.assert_equal(test,expectDict['test_withCategory'])

    def test_lag_collection_withTime(self,data_input_collection,expected_lag_output_collection):
        df_train = pd.DataFrame(data_input_collection['train'])
        df_test = pd.DataFrame(data_input_collection['test'])
        expectDict = {}
        for i in expected_lag_output_collection:
            expectDict[i]=pd.DataFrame(expected_lag_output_collection[i]).values
        transformer = Lag_Transformer(1,'time','category',remove_time=False)
        transformer = transformer.fit(df_train)
        train = transformer.transform(df_train)
        test = transformer.transform(df_test)
        np.testing.assert_equal(train,expectDict['train_withTime'])
        np.testing.assert_equal(test,expectDict['test_withTime'])

    def test_lag_collection_withTimeCategory(self,data_input_collection,expected_lag_output_collection):
        df_train = pd.DataFrame(data_input_collection['train'])
        df_test = pd.DataFrame(data_input_collection['test'])
        expectDict = {}
        for i in expected_lag_output_collection:
            expectDict[i]=pd.DataFrame(expected_lag_output_collection[i]).values
        transformer = Lag_Transformer(1,'time','category',remove_time=False,remove_category=False)
        transformer = transformer.fit(df_train)
        train = transformer.transform(df_train)
        test = transformer.transform(df_test)
        np.testing.assert_equal(train,expectDict['train_withTimeCategory'])
        np.testing.assert_equal(test,expectDict['test_withTimeCategory'])

    def test_func_single(self,data_input_single,expected_func_output_single):
        def add_x(data,x):
            res = collections.defaultdict(list)
            for i in data:
                for v in data[i]:
                    res[i].append(v+x)
            return pd.DataFrame(res)
        func = add_x
        df_train = pd.DataFrame(data_input_single['train'])
        numpyData_train = df_train.values
        expectDict = {}
        for i in expected_func_output_single:
            expectDict[i]=pd.DataFrame(expected_func_output_single[i]).values
        df_test = pd.DataFrame(data_input_single['test'])
        numpyData_test = df_test.values
        transformer = Function_Transformer(func,['data1','data2'],'time',parameterDict={'x':1})
        transformer = transformer.fit(df_train)
        train = transformer.transform(df_train)
        test = transformer.transform(df_test)
        np.testing.assert_equal(train,expectDict['train_rmTime'])
        np.testing.assert_equal(test,expectDict['test_rmTime'])
        transformer = Function_Transformer(func,['data1','data2'],'time',remove_time=False,parameterDict={'x':1})
        transformer = transformer.fit(df_train)
        train = transformer.transform(df_train)
        test = transformer.transform(df_test)
        np.testing.assert_equal(train,expectDict['train_withTime'])
        np.testing.assert_equal(test,expectDict['test_withTime'])

##########
    def test_func_collection(self,data_input_collection,expected_func_output_collection):
        def add_x(data,x):
            res = collections.defaultdict(list)
            for i in data:
                for v in data[i]:
                    res[i].append(v+x)
            return pd.DataFrame(res)
        func = add_x
        df_train = pd.DataFrame(data_input_collection['train'])
        df_test = pd.DataFrame(data_input_collection['test'])
        expectDict = {}
        for i in expected_func_output_collection:
            expectDict[i]=pd.DataFrame(expected_func_output_collection[i]).values
        transformer = Function_Transformer(func,['data'],'time','category',parameterDict={'x':1})
        transformer = transformer.fit(df_train)
        train = transformer.transform(df_train)
        test = transformer.transform(df_test)
        np.testing.assert_equal(train,expectDict['train_withoutTimeCategory'])
        np.testing.assert_equal(test,expectDict['test_withoutTimeCategory'])


    def test_func_collection_withCategory(self,data_input_collection,expected_func_output_collection):
        def add_x(data,x):
            res = collections.defaultdict(list)
            for i in data:
                for v in data[i]:
                    res[i].append(v+x)
            return pd.DataFrame(res)
        func = add_x

        df_train = pd.DataFrame(data_input_collection['train'])
        df_test = pd.DataFrame(data_input_collection['test'])
        expectDict = {}
        for i in expected_func_output_collection:
            expectDict[i]=pd.DataFrame(expected_func_output_collection[i]).values
        transformer = Function_Transformer(func,['data'],'time','category',remove_category=False,parameterDict={'x':1})
        transformer = transformer.fit(df_train)
        train = transformer.transform(df_train)
        test = transformer.transform(df_test)
        np.testing.assert_equal(train,expectDict['train_withCategory'])
        np.testing.assert_equal(test,expectDict['test_withCategory'])
#
    def test_func_collection_withTime(self,data_input_collection,expected_func_output_collection):
        def add_x(data,x):
            res = collections.defaultdict(list)
            for i in data:
                for v in data[i]:
                    res[i].append(v+x)
            return pd.DataFrame(res)
        func = add_x
        df_train = pd.DataFrame(data_input_collection['train'])
        df_test = pd.DataFrame(data_input_collection['test'])
        expectDict = {}
        for i in expected_func_output_collection:
            expectDict[i]=pd.DataFrame(expected_func_output_collection[i]).values
        transformer = Function_Transformer(func,['data'],'time','category',remove_time=False,parameterDict={'x':1})
        transformer = transformer.fit(df_train)
        train = transformer.transform(df_train)
        test = transformer.transform(df_test)
        np.testing.assert_equal(train,expectDict['train_withTime'])
        np.testing.assert_equal(test,expectDict['test_withTime'])

    def test_func_collection_withTimeCategory(self,data_input_collection,expected_func_output_collection):
        def add_x(data,x):
            res = collections.defaultdict(list)
            for i in data:
                for v in data[i]:
                    res[i].append(v+x)
            return pd.DataFrame(res)
        func = add_x
        df_train = pd.DataFrame(data_input_collection['train'])
        df_test = pd.DataFrame(data_input_collection['test'])
        expectDict = {}
        for i in expected_func_output_collection:
            expectDict[i]=pd.DataFrame(expected_func_output_collection[i]).values
        transformer = Function_Transformer(func,['data'],'time','category',remove_time=False,remove_category=False,parameterDict={'x':1})
        transformer = transformer.fit(df_train)
        train = transformer.transform(df_train)
        test = transformer.transform(df_test)
        np.testing.assert_equal(train,expectDict['train_withTimeCategory'])
        np.testing.assert_equal(test,expectDict['test_withTimeCategory'])

    def test_single_stock_technical_indicator(self,single_stock):
        strategy = ta.Strategy(
            name = 'sma',
            ta = [
                {'kind':'sma','length':2}
            ]
        )
        df = pd.DataFrame(single_stock['train'])
        sit = Stock_Technical_Indicator_Transformer(strategy,'Date')
        sit.fit(df)
        y = sit.transform(df)
        df.ta.strategy(strategy)
        np.testing.assert_equal(y.reshape(-1),df['SMA_2'].values)
        df = pd.DataFrame(single_stock['train'])
        df = df.append(pd.DataFrame(single_stock['test']),True)
        df2 = pd.DataFrame(single_stock['test'])
        y = sit.transform(df2)
        df.ta.strategy(strategy)
        np.testing.assert_equal(y.reshape(-1),df.tail(3)['SMA_2'].values)

    def test_collection_stock_technical_indicator(self,collection_stock):
        strategy = ta.Strategy(
            name = 'sma',
            ta = [
                {'kind':'sma','length':2}
            ]
        )
        df = pd.DataFrame(collection_stock['train'])
        sit = Stock_Technical_Indicator_Transformer(strategy,'Date','symbol')
        sit.fit(df)
        y = sit.transform(df)
        res = []
        for i in df.symbol.unique():
            tmp = df[df.symbol == i]
            tmp.ta.strategy(strategy)
            res.extend(tmp['SMA_2'])
        np.testing.assert_equal(y.reshape(-1),np.array(res))
        df = pd.DataFrame(collection_stock['test'])
        y= sit.transform(df)
        df = pd.DataFrame(collection_stock['train'])
        df = df.append(pd.DataFrame(collection_stock['test']),True)
        res = []
        for i in df.symbol.unique():
            tmp = df[df.symbol == i]
            tmp.ta.strategy(strategy)
            res.extend(tmp['SMA_2'].tail(2).tolist())
        np.testing.assert_equal(y.reshape(-1),np.array(res))