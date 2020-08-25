import os
import pytest
import numpy as np
import pandas as pd
from time_series_transform.transform_core_api.base import *



class Time_Series_Base:
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
        assert tsd[:,['d1']] == {'d1':[4,5,6],'time':[1,2,3]}
        assert tsd[:,['d1','l1']] == {'d1':[4,5,6],'l1':['a','b','c'],'time':[1,2,3]}

    def test_time_series_base_sort(self):
        tsd = Time_Series_Data()
        tsd.set_time_index([1,3,2],'time')
        tsd.set_data([4,5,6],'d1')
        assert tsd.sort(True)[:,['d1']] == [4,6,5]
        assert tsd.sort(False)[:,['d1']] == [5,6,4]

    def test_time_series_base_make_frame(self):
        compareDf = pd.DataFrame({
            'time':[1,3,2],
            'd1':[4,5,6]
        })
        tsd = Time_Series_Data()
        tsd.set_time_index([1,3,2],'time')
        tsd.set_data([4,5,6],'d1')
        assert tsd.make_dataframe().equals(compareDf)

        