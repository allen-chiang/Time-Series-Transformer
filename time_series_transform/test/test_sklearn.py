import os
import copy
import pytest
import numpy as np
import pandas as pd
from time_series_transform.sklearn.transformer import (
    Base_Time_Series_Transformer,
    Lag_Transformer,
    Lead_Transformer,
    Function_Transformer
)

@pytest.fixture('class')
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

@pytest.fixture('class')
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


class Test_sklearn_transformer:

    def test_base_input_single(self,data_input_single):
        df = pd.DataFrame(data_input_single['train'])
        numpyData = df.values
        transformer = Base_Time_Series_Transformer('time',None,'ignore')
        transformer = transformer.fit(df)
        print(transformer.get_time_series_cache())
        assert transformer.get_time_series_cache() == df.time.tolist()
        transformer = Base_Time_Series_Transformer(0,None,'ignore')
        transformer = transformer.fit(numpyData)
        assert transformer.get_time_series_cache() == df.time.tolist()


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


    def test_lead_single(self):
        pass
