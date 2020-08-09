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
    params=[
        {'newIX':True,'byCategory':False},
        {'newIX':True,'byCategory':True}
        ]
    )
def expanded_transformer(request):
    data = {
        'time':[1,2,3,4,4],
        'positive_float': [1.0,2.0,3.0,4.0,4.0], 
        'negative_float': [-1.0,-2.0,-3.0,-4.0,-4.0], 
        'positive_int':[1,2,3,4,4], 
        'negative_int':[-1,-2,-3,-4,-4], 
        'nan':[np.nan,np.nan,np.nan,np.nan,np.nan],
        'category':[1,1,1,2,1]
    }
    df = pd.DataFrame(data)
    tst = Pandas_Time_Series_Tensor_Dataset(df)
    tst.expand_dataFrame_by_date(
        'category',
        'time',
        newIX = request.param['newIX'],
        byCategory= request.param['byCategory'],
        dropna=False
        )    
    return tst,request.param['byCategory']

class Test_pandas_to_tensor:
    
    @pytest.mark.parametrize(
        "cases", 
        [
            {'newIX':True,'byCategory':True,'dropna':False},
            {'newIX':True,'byCategory':False,'dropna':False},
            {'newIX':True,'byCategory':True,'dropna':True},
            {'newIX':True,'byCategory':False,'dropna':True},
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
                      
    def test_time_series_tensor_stack_lags(self,expanded_transformer):
        tst,byCategory = expanded_transformer
        if byCategory == True:
            colList = ["positive_float","negative_float","nan"]
        else:
            colList = ["1_positive_float","1_negative_float","1_nan"]
        for ix,v in enumerate(colList):
            if ix == 0:
                tst.set_config(
                    'stackLag',
                    [f"{v}_{i}"for i in range(1,5)],
                    'sequence',
                    None,
                    False,
                    2,
                    0,
                    np.float32
                    )
            else:
                tst.set_config(
                    f'stackLag_{v}',
                    [f"{v}_{i}"for i in range(1,5)],
                    'sequence',
                    "stackLag",
                    False,
                    2,
                    0,
                    np.float32
                    )
        gen = tst.make_data_generator()
        for i in gen:
            print(i[0])
            assert i[0]['stackLag'].shape == (2,2,len(colList))
        

    # def test_time_series_tensor_stack_leads(self,expanded_transformer):
    #     tst,byCategory = expanded_transformer
    #     if byCategory == True:
    #         colList = ["positive_float","negative_float","nan"]
    #     else:
    #         colList = ["1_positive_float","1_negative_float","1_nan"]
    #     for ix,v in enumerate(colList):
    #         if ix == 0:
    #             tst.set_config(
    #                 'stackLag',
    #                 [f"{v}_{i}"for i in range(1,5)],
    #                 'sequence',
    #                 None,
    #                 False,
    #                 2,
    #                 0,
    #                 np.float32
    #                 )
    #         else:
    #             tst.set_config(
    #                 f'stackLag_{v}',
    #                 [f"{v}_{i}"for i in range(1,5)],
    #                 'sequence',
    #                 "stackLag",
    #                 False,
    #                 2,
    #                 0,
    #                 np.float32
    #                 )
    #     gen = tst.make_data_generator()
    #     for i in gen:
    #         print(i[0])
    #         assert i[0]['stackLag'].shape == (2,2,len(colList))

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


class Test_time_series_util:
    def test_rollilng_window(self):
        pass

    def test_rolling_identity(self):
        pass

    def test_moving_average(self):
        pass

    def teest_rfft_transform(self):
        pass

    def test_wavelet_denoising(self):
        pass