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
    return [{
        'time': [1,2,3,1,2,3],
        'data':[1,2,3,1,2,3],
        'category':[1,1,1,2,2,2]
    },
        {
        'time': [],
        'data':[],
        'category':[]
    }]

@pytest.fixture('class')
def expect_single_noChange():
    return {
            'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'data': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }

@pytest.fixture('class')
def expect_single_expandTime():
    return {
            'data_1': [1],
            'data_2': [2],
            'data_3': [3],
            'data_4': [4],
            'data_5': [5],
            'data_6': [6],
            'data_7': [7],
            'data_8': [8],
            'data_9': [9],
            'data_10': [10],
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

    def test_base_io_from_collection(self, dictList_collection):
        noChange = dictList_collection
        expandCategory = []
        fullExpand = []
        data = dictList_collection[0]
        ts = Time_Series_Data()
        ts = ts.set_time_index(data['time'], 'time')
        ts = ts.set_data(data['data'], 'data')
        ts = ts.set_data(data['category'],'category')
        tsc = Time_Series_Data_Collection(ts,'time','category')
        io = io_base(tsc, 'time', 'category')
        timeSeries = io.from_collection(False,True)
        expandTime =[]
        for i in timeSeries:
            assert timeSeries[i] == expandTime[0][i]

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
