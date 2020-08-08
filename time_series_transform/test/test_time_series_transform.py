import pytest
import numpy as np
import pandas as pd
from time_series_transform.transform_core_api import *
from time_series_transform.transform_core_api.util import *
from time_series_transform.transform_core_api.base import *


@pytest.fixture(scope='class')
def single_category_dataframe():
    data = {
        'positive_float': [1.0,2.0,3.0,4.0],
        'negative_float': [-1.0,-2.0,-3.0,-4.0],
        'positive_int':[1,2,3,4],
        'negative_int':[-1,-2,-3,-4],
        'nan':[np.nan,np.nan,np.nan,np.nan],
        'category':['1','1','2','2']
    }
    return pd.DataFrame(data)

@pytest.fixture(scope='class')
def two_category_dataframe():
    data = {
        'positive_float': [1.0,2.0,3.0,4.0],
        'negative_float': [-1.0,-2.0,-3.0,-4.0],
        'positive_int':[1,2,3,4],
        'negative_int':[-1,-2,-3,-4],
        'nan':[np.nan,np.nan,np.nan,np.nan],
        'category1':['1','1','2','2']
        'category2':['3','3','4','4']
    }
    return pd.DataFrame(data)


class test_pandas_to_tensor:
    def test_time_series_tensor_stack_lags(self):
        pass


    def test_time_series_tensor_stack_leads(self):
        pass


    def test_time_series_tensor_category(self):
        pass

    def test_time_series_dataset_make_dataset(self):
        pass


    def test_pandas_to_tensor_expand_date(self):
        pass


    def test_pandas_to_tensor_make_generator(self):
        pass



class test_pandas_to_time_panel:
    def test_pandas_panel_expand_category(self):
        pass

    def test_pandas_panel_make_lags(self):
        pass


    def test_pandas_panel_make_leads(self):
        pass


    def test_make_transformation(self):
        pass