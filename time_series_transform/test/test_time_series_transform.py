import copy
import pytest
import numpy as np
import pandas as pd
from time_series_transform.transform_core_api import *
from time_series_transform.transform_core_api.util import *
from time_series_transform.transform_core_api.base import *
from time_series_transform.transform_core_api.time_series_transformer import *



@pytest.fixture('class')
def dictList_single():
    return {
        'time': [1, 2],
        'data': [1, 2]
    }

@pytest.fixture('class')
def dictList_collection():
    return {
        'time': [1,2,1,3],
        'data':[1,2,1,2],
        'category':[1,1,2,2]
    }
    
@pytest.fixture('class')
def expect_single_lag():
    return {
        'time':[1,2],
        'data':[1,2],
        'data_lag_1':[np.nan,1]
    }

@pytest.fixture('class')
def expect_single_lead():
    return {
        'time':[1,2],
        'data':[1,2],
        'data_lead_1':[2,np.nan]
    }

@pytest.fixture('class')
def expect_single_lag_sequence():
    return {
        'time':[1,2],
        'data':[1,2],
        'data_lag_1':[[np.nan],[1]]
    }

@pytest.fixture('class')
def expect_single_identical():
    return {
        
    }

# To-do
# Test
#   io input
#   io output with label or without label
#   transformation
#       lag, sequence, lead, identical sequence


class Test_time_series_transform:
    def test_from_pandas(self):
        pass

    def test_to_pandas(self):
        pass

    def test_from_numpy(self):
        pass

    def test_to_numpy(self):
        pass

    def test_from_dict(self):
        pass

    def test_to_dict(self):
        pass

    def test_to_tfDataset(self):
        pass

    def test_single_make_lag(self,dictList_single,expect_single_lag):
        data = dictList_single
        expectDf = pd.DataFrame(expect_single_lag)
        tst = Time_Series_Transformer(data,'time',None)
        tst.make_lag('data',1,'_lag_')
        df = tst.to_pandas()
        pd.testing.assert_frame_equal(df,expectDf,False)

    def test_single_make_lead(self,dictList_single,expect_single_lead):
        data = dictList_single
        expectDf = pd.DataFrame(expect_single_lead)
        tst = Time_Series_Transformer(data,'time',None)
        tst.make_lead('data',1,'_lead_')
        df = tst.to_pandas()
        pd.testing.assert_frame_equal(df,expectDf,False)

    def test_single_lag_sequence(self,dictList_single,expect_single_lag_sequence):
        data = dictList_single
        expectDf = pd.DataFrame(expect_single_lag_sequence)
        tst = Time_Series_Transformer(data,'time',None)
        tst.make_lag_sequence('data',1,1,'_lag_')
        print(tst.time_series_data)
        df = tst.to_pandas()
        pd.testing.assert_frame_equal(df,expectDf,False)


    def test_single_sequence(self):
        pass


    def test_transform(self):
        pass

    def test_make_sequence(self):
        pass