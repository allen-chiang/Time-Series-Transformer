import copy
import pytest
import numpy as np
import pandas as pd
from time_series_transform.io.base import io_base
from time_series_transform.io.numpy import (from_numpy, to_numpy)
from time_series_transform.io.pandas import (from_pandas, to_pandas)
from time_series_transform.transform_core_api.base import (
    Time_Series_Data, Time_Series_Data_Collection)


@pytest.fixture('class')
def dictList_single():
    return {
        'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'data': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }

@pytest.fixture('class')
def dictList_collection():
    return [
        {
        'time': [1,2,1,3],
        'data':[1,2,1,2],
        'category':[1,1,2,2]
    }]

@pytest.fixture('class')
def expect_collection_expandTime():
    return {
        'pad': {
            'data1':[1,1],
            'data2':[2,np.nan],
            'data3':[np.nan,2],
            'category':[1,2]
        },
        'remove': {
            'data1':[1,1],
            'category':[1,2]            
        }
    }
@pytest.fixture('class')
def expect_collection_expandCategory():
    return {
        'pad': {
            'time':[1,2,3],
            'data1':[1,2,np.nan],
            'data2':[1,np.nan,2]
        },
        'remove': {
            'time':[1],
            'data1':[1],
            'data2':[1]            
        }        
    }

@pytest.fixture('class')
def expect_collection_expandFull():
    return {
        'pad': {
            'data_1_1':[1],
            'data_1_2':[2],
            'data_1_3':[np.nan],
            'data_2_1':[1],
            'data_2_2':[np.nan],
            'data_2_3':[2]
        },
        'remove': {
            'data_1_1':[1],
            'data_2_1':[1], 
        }        
    }


class Test_base_io:

    def test_base_io_from_single(self, dictList_single,expect_single_noChange,expect_single_expandTime):
        NoExpandTimeAns = expect_single_noChange
        ExpandTimeAns = expect_single_expandTime
        data = dictList_single
        ts = Time_Series_Data()
        ts = ts.set_time_index(data['time'], 'time')
        ts = ts.set_data(data['data'], 'data')
        io = io_base(ts, 'time', None)
        timeSeries = io.from_single(False)
        for i in timeSeries:
            assert timeSeries[i].tolist() == NoExpandTimeAns[i]
        timeSeries = io.from_single(True)
        for i in timeSeries:
            assert timeSeries[i] == ExpandTimeAns[i]

    def test_base_io_to_single(self, dictList_single):
        data = dictList_single
        ts = Time_Series_Data()
        ts = ts.set_time_index(data['time'], 'time')
        ts = ts.set_data(data['data'], 'data')
        io = io_base(data, 'time', None)
        assert io.to_single(data, 'time') == ts

    def test_base_io_from_collection_expandTime(self, dictList_collection,expect_single_expandTime):
        noChange = dictList_collection
        expand = expect_single_expandTime
        fullExpand = []
        data = dictList_collection[0]
        ts = Time_Series_Data()
        ts = ts.set_time_index(data['time'], 'time')
        ts = ts.set_data(data['data'], 'data')
        ts = ts.set_data(data['category'],'category')
        tsc = Time_Series_Data_Collection(ts,'time','category')
        io = io_base(tsc, 'time', 'category')
        with pytest.raises(ValueError):
            timeSeries = io.from_collection(False,True,'ignore')
        timeSeries = io.from_collection(False,True,'pad')
        for i in timeSeries:
            assert timeSeries[i] == expand['pad'][i]
        timeSeries = io.from_collection(False,True,'remove')
        for i in timeSeries:
            assert timeSeries[i] == expand['remove'][i]

    def test_base_io_from_collection_expandCategory(self, dictList_collection,expect_single_expandCategory):
        noChange = dictList_collection
        expand = expect_single_expandCategory
        fullExpand = []
        data = dictList_collection[0]
        ts = Time_Series_Data()
        ts = ts.set_time_index(data['time'], 'time')
        ts = ts.set_data(data['data'], 'data')
        ts = ts.set_data(data['category'],'category')
        tsc = Time_Series_Data_Collection(ts,'time','category')
        io = io_base(tsc, 'time', 'category')
        with pytest.raises(ValueError):
            timeSeries = io.from_collection(True,False,'ignore')
        timeSeries = io.from_collection(True,False,'pad')
        for i in timeSeries:
            assert timeSeries[i] == expand['pad'][i]
        timeSeries = io.from_collection(True,False,'remove')
        for i in timeSeries:
            assert timeSeries[i] == expand['remove'][i]

    def test_base_io_from_collection_expandFull(self, dictList_collection,expect_single_expandFull):
        noChange = dictList_collection
        expand = expect_single_expandFull
        fullExpand = []
        data = dictList_collection[0]
        ts = Time_Series_Data()
        ts = ts.set_time_index(data['time'], 'time')
        ts = ts.set_data(data['data'], 'data')
        ts = ts.set_data(data['category'],'category')
        tsc = Time_Series_Data_Collection(ts,'time','category')
        io = io_base(tsc, 'time', 'category')
        with pytest.raises(ValueError):
            timeSeries = io.from_collection(True,True,'ignore')
        timeSeries = io.from_collection(True,True,'pad')
        for i in timeSeries:
            assert timeSeries[i] == expand['pad'][i]
        timeSeries = io.from_collection(True,True,'remove')
        for i in timeSeries:
            assert timeSeries[i] == expand['remove'][i]

    def test_base_io_to_collection(self, dictList_collection):
        dataList = dictList_collection
        pass


class Test_Pandas_IO:
    def test_from_pandas_single(self):
        pass

    def test_to_pandas_single(self):
        pass

    def test_from_pandas_collection(self):
        pass

    def test_to_pandas_collection(self):
        pass


class Test_Numpy_IO:
    def test_from_numpy_single(self):
        pass

    def test_to_numpy_single(self):
        pass

    def test_from_numpy_collection(self):
        pass

    def test_to_numpy_collection(self):
        pass

class Test_Generator_IO:
    def test_from_generator(self):
        pass

    def test_to_generator(self):
        pass