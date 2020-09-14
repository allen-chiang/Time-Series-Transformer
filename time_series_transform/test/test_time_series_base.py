import os
import pytest
import numpy as np
import pandas as pd
import copy
from time_series_transform.transform_core_api.base import *


class Test_time_series_base:

    def test_time_series_base_length(self):
        tsd = Time_Series_Data()
        tsd.set_time_index([1,2,3],'time')
        
        with pytest.raises(ValueError):
            tsd.set_data([1],'data1')
            tsd.set_labels([3],'label')

    def test_time_series_base_slice(self):
        tsd = Time_Series_Data()
        tsd.set_time_index([1,2,3],'time')
        tsd.set_data([4,5,6],'d1')
        tsd.set_labels(['a','b','c'],'l1')
        np.testing.assert_array_equal(tsd[:,['d1']]['d1'] ,np.array([4,5,6]))

    def test_time_series_base_sort(self):
        tsd = Time_Series_Data()
        tsd.set_time_index([1,3,2],'time')
        tsd.set_data([4,5,6],'d1')
        np.testing.assert_array_equal(tsd.sort(True)[:,['d1']]['d1'],np.array([4,6,5]))
        np.testing.assert_array_equal(tsd.sort(False)[:,['d1']]['d1'],np.array([5,6,4]))

    def test_time_series_base_make_frame(self):
        compareDf = pd.DataFrame({
            'time':np.array([1,3,2]),
            'd1':np.array([4,5,6])
        })
        tsd = Time_Series_Data()
        tsd.set_time_index([1,3,2],'time')
        tsd.set_data([4,5,6],'d1')
        assert tsd.make_dataframe().equals(compareDf)

    def test_time_series_base_transform(self):
        tsd = Time_Series_Data()
        tsd.set_time_index([1,3,2],'time')
        tsd.set_data([4,5,6],'d1')
        tsd.sort()
        tsd.transform('d1','res',lambda x: x*2)
        np.testing.assert_array_equal(tsd[:,['res']]['res'] , np.array([8,12,10]))
        tsd.transform('d1','res',lambda x: pd.Series(x*2))
        np.testing.assert_array_equal(tsd[:,['res']]['res'] , np.array([8,12,10]))
        tsd.transform('d1','res',lambda x: pd.DataFrame({'res':x*2}))
        np.testing.assert_array_equal(tsd[:,['res_res']]['res_res'] , np.array([8,12,10]))


@pytest.fixture(scope = 'class')
def data():
    data = {
        'data1':[1,2,1,2],
        'data2':["1","2","1","2"],
        'label':["2",'2','a','a'],
        'time':[1,2,1,2]
    }
    tsd = Time_Series_Data()
    tsd.set_time_index(data['time'],'time')
    tsd.set_data(data['data1'],'data1')
    tsd.set_data(data['data2'],'data2')
    tsd.set_labels(data['label'],"label")
    return tsd

@pytest.fixture(scope = 'class')
def data_different_date():
    data = {
        'data1':[1,2,1,2],
        'data2':["1","2","1","2"],
        'label':["2",'2','a','a'],
        'time':[1,2,1,3]
    }
    tsd = Time_Series_Data()
    tsd.set_time_index(data['time'],'time')
    tsd.set_data(data['data1'],'data1')
    tsd.set_data(data['data2'],'data2')
    tsd.set_labels(data['label'],"label")
    return tsd

class Test_Time_Series_Collection:
    
    def test_time_series_collection_slice(self,data):
        data = data
        tsdc = Time_Series_Data_Colleciton(data,'time','label')
        assert tsdc["a"] == tsdc["2"]
        
    def test_time_series_collection_transform(self,data):
        def add_one(arr):
            res = []
            for a in arr:
                res.append(a+1)
            return res
        data = data
        tsdc = Time_Series_Data_Colleciton(data,'time','label')
        check1 = copy.deepcopy(tsdc['a'])
        check1.transform('data1','data1',add_one)
        tsdc.transform('data1','data1',add_one)
        assert check1 == tsdc['2'] == tsdc['a']

    def test_time_series_collection_remove_different_date(self,data_different_date):
        data = data_different_date
        tsdc = Time_Series_Data_Colleciton(data,'time','label')
        tsdc=tsdc.remove_different_time_index()
        assert tsdc['a'].time_index['time'] == [1] == tsdc['2'].time_index['time']

    def test_time_series_collection_padding_date(self,data_different_date):
        data = data_different_date
        tsdc = Time_Series_Data_Colleciton(data,'time','label')
        tsdc=tsdc.pad_time_index()
        assert list(tsdc['a'].time_index['time']) == list(tsdc['2'].time_index['time']) == [1,2,3]

    def test_time_series_collection_sort(self,data_different_date):
        data = data_different_date
        tsdc = Time_Series_Data_Colleciton(data,'time','label')
        tsdc = tsdc.pad_time_index()
        tsdc = tsdc.sort(False,['2'])
        tsdc = tsdc.sort(True,['a'])
        assert list(tsdc['a'].time_index['time']) == [1,2,3]
        assert list(tsdc['2'].time_index['time']) == [3,2,1]

    def test_time_series_transform_make_dataframe(self,data):
        data = data
        tsdc = Time_Series_Data_Colleciton(data,'time','label')
        test1 = tsdc.make_dataframe(False,False)
        check2 = data.make_dataframe()
        assert np.array_equal(test1[test1.columns].sort_values('label').values,check2[test1.columns].sort_values('label').values)
        test2 = tsdc.make_dataframe(True,False)
        check2 = pd.DataFrame({'time':np.array([1,2],np.int32),
                    'data1_a':np.array([1,2],np.int32),
                    'data2_a':np.array(['1','2'],object),
                    'data1_2':np.array([1,2],np.int32),
                    'data2_2':np.array(['1','2'],object)}
                  )
        assert np.array_equal(test2.values,check2)
        test3 = tsdc.make_dataframe(True,True)
        check3 =  pd.DataFrame({'data1_a_1': [1],
                            'data2_a_1': ['1'],
                            'data1_2_1': [1],
                            'data2_2_1': ['1'],
                            'data1_a_2': [2],
                            'data2_a_2': ['2'],
                            'data1_2_2': [2],
                            'data2_2_2': ['2']})
        assert np.array_equal(test3.values,check3)