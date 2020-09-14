import os
import pytest
import numpy as np
import pandas as pd
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


class Test_Time_Series_Collection:

    def test_time_series_collection_slice(self):
        # test ssh key
        pass

    def test_time_series_collection_transform(self):
        pass

    def test_time_series_collection_remove_different_date(self):
        pass

    def test_time_series_collection_padding_date(self):
        pass

    def test_time_series_collection_sort(self):
        pass

    def test_time_series_transform_make_dataframe(self):
        pass