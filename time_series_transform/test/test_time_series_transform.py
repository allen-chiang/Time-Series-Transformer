import pytest
import numpy as np
import pandas as pd
from time_series_transform.transform_core_api import *
from time_series_transform.transform_core_api.util import *
from time_series_transform.transform_core_api.base import *
from time_series_transform.transform_core_api.time_series_transformer import *

@pytest.fixture(scope='class')
def category_dataframe():
    data = {
        'time':[1,2,3,4],
        'positive_float': [1.0,2.0,3.0,4.0], 
        'negative_float': [-1.0,-2.0,-3.0,-4.0], 
        'positive_int':[1,2,3,4], 
        'negative_int':[-1,-2,-3,-4], 
        'nan':[np.nan,np.nan,np.nan,np.nan],
        'category':[1,1,2,2]
    }
    return pd.DataFrame(data)

@pytest.fixture(
    scope='class',
    params=[]
    )
def expanded_transformer(request):
    data = {
        'time':[1,2,3,4],
        'positive_float': [1.0,2.0,3.0,4.0], 
        'negative_float': [-1.0,-2.0,-3.0,-4.0], 
        'positive_int':[1,2,3,4], 
        'negative_int':[-1,-2,-3,-4], 
        'nan':[np.nan,np.nan,np.nan,np.nan],
        'category':[1,1,2,2]
    }
    df = pd.DataFrame(data)
    tst = Pandas_Time_Series_Tensor_Dataset(df)
    tst.expand_dataFrame_by_date(
        'category',
        'time',
        newIX = cases['newIX'],
        byCategory= cases['byCategory'],
        dropna=False
        )    
    return tst


class Test_pandas_to_tensor:
    
    @pytest.mark.parametrize(
        "cases", 
        [
            {'newIX':True,'byCategory':True,'dropna':False},
            {'newIX':False,'byCategory':False,'dropna':False},
            {'newIX':True,'byCategory':False,'dropna':False},
            {'newIX':False,'byCategory':True,'dropna':False},
            {'newIX':True,'byCategory':True,'dropna':True},
            {'newIX':True,'byCategory':True,'dropna':False},
            ]
        )
    def test_pandas_to_tensor_expand_date(self,cases,category_dataframe):
        df = category_dataframe
        tst = Pandas_Time_Series_Tensor_Dataset(df)
        tst.expand_dataFrame_by_date(
            'category',
            'time',
            newIX = cases['newIX'],
            byCategory= cases['byCategory'],
            dropna=False
            )
        if cases['byCategory'] == False:
            shapes = (1,(df.shape[1]-2)*len(df))
        else:
            shapes = (len(df.category.unique()),(df.shape[1]-2)*len(df))            
        assert tst.df.shape == shapes
        if cases['dropna'] == True:
            tst = Pandas_Time_Series_Tensor_Dataset(df)
            tst.expand_dataFrame_by_date(
                'category',
                'time',
                **cases
                )  
            assert tst.df.isna().sum().sum() == 0
                      

    @pytest.mark.dependency(depends=["test_pandas_to_tensor_expand_date"])
    def test_time_series_tensor_stack_lags(self,category_dataframe):
        pass

    # @pytest.mark.dependency(depends=["test_pandas_to_tensor_expand_date"])
    # def test_time_series_tensor_stack_leads(self):
    #     pass

    # @pytest.mark.dependency(depends=["test_pandas_to_tensor_expand_date"])
    # def test_time_series_tensor_category(self):
    #     pass

    # @pytest.mark.dependency(depends=["test_pandas_to_tensor_expand_date"])
    # def test_time_series_dataset_make_dataset(self):
    #     pass

    # @pytest.mark.dependency(depends=["test_pandas_to_tensor_expand_date"])
    # def test_pandas_to_tensor_make_generator(self):
    #     pass

# class Test_pandas_to_time_panel:
#     def test_pandas_panel_expand_category(self):
#         pass

#     def test_pandas_panel_make_lags(self):
#         pass


#     def test_pandas_panel_make_leads(self):
#         pass


#     def test_make_transformation(self):
#         pass