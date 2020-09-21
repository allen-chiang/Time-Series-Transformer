import copy
import pytest
import numpy as np
import pandas as pd
from time_series_transform.io.base import io_base
from time_series_transform.io.numpy import (from_numpy,to_numpy)
from time_series_transform.io.pandas import (from_pandas,to_pandas)
from time_series_transform.transform_core_api.base import (Time_Series_Data,Time_Series_Data_Collection)


@pytest.fixture('class')
def dictList_single ():
    return {
        'time':[1,2,3,4,5,6,7,8,9,10],
        'data':[1,2,3,4,5,6,7,8,9,10]
    }

@pytest.fixture('class')
def dictList_collection ():
    return [{
        'time':[1,2,3,4,5,1,2,3,4,10],
        'data':[1,2,3,4,5,6,7,8,9,10],
        'category':[1,1,1,1,1,2,2,2,2,2]
    },
    {
        'time':[1,2,3,4,5,1,2,3,4,5],
        'data':[1,2,3,4,5,6,7,8,9,10],
        'category':[1,1,1,1,1,2,2,2,2,2]
    }
    ]

class Test_base_io:

    def test_base_io_from_single(self,dictList_single):
        NoExpandTimeAns = {
            'time':[1,2,3,4,5,6,7,8,9,10],
            'data':[1,2,3,4,5,6,7,8,9,10]
        }
        ExpandTimeAns = {
            'data_1':[1],
            'data_2':[2],
            'data_3':[3],
            'data_4':[4],
            'data_5':[5],
            'data_6':[6],
            'data_7':[7],
            'data_8':[8],
            'data_9':[9],            
            'data_10':[10],
        }
        data = dictList_single
        ts = Time_Series_Data()
        ts = ts.set_time_index(data['time'],'time')
        ts = ts.set_data(data['data'],'data')
        io = io_base(ts,'time',None)
        timeSeries = io.from_single(False)
        for i in timeSeries:
            assert timeSeries[i].tolist() == NoExpandTimeAns[i]
        timeSeries = io.from_single(True)
        for i in timeSeries:
            assert timeSeries[i] == ExpandTimeAns[i]


    def test_base_io_to_single(self):
        pass

    def test_base_io_from_collection(self):
        pass

    def test_base_io_to_collection(self):
        pass


