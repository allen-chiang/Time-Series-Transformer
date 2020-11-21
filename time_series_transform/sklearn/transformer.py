import numpy as np
import pandas as pd
from time_series_transform.transform_core_api.time_series_transformer import Time_Series_Transformer
from sklearn.base import (BaseEstimator, TransformerMixin)



class Base_Time_Series_Transformer(BaseEstimator, TransformerMixin):

    def __init__(self,time_col,category_col=None,len_preprocessing = 'ignore',remove_time=True,cache_data_path = None):
        self._time_col = time_col 
        self._category_col = category_col
        self._time_series_cache = None
        self._len_preprocessing = len_preprocessing
        self._remove_time = remove_time
        self.cache_data_path = cache_data_path
 
    def fit( self, X, y = None ):
        if isinstance(X, pd.DataFrame):
            self._time_series_cache = X[self._time_col].tolist()
            return self
        self._time_series_cache = list(X[:,self._time_col])
        return self 

    def transform( self, X, y = None ):
        if isinstance(X,pd.DataFrame):
            return X.values
        return X

    def get_time_series_cache (self):
        return self._time_series_cache


class Lag_Transformer(Base_Time_Series_Transformer):
    def __init__(self,lag_num,time_col,category_col=None,len_preprocessing = 'ignore',remove_time = True,cache_data_path=None):
        super().__init__(time_col,category_col,len_preprocessing,remove_time,cache_data_path)
        self._X = None
        self.time_series_transform = None
        self.lag_num = lag_num

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        return self

class Lead_Transformer(Base_Time_Series_Transformer):
    def __init__(self,time_col,category_col=None,len_preprocessing = 'ignore'):
        super().__init__(time_col,category_col,len_preprocessing)

    def fit(self):
        return self

    def transform(self):
        return self

class Function_Transformer(Base_Time_Series_Transformer):
    def __init__(self,func,time_col,category_col=None,len_preprocessing = 'ignore'):
        super().__init__(time_col,category_col,len_preprocessing)
        self._func = func
        
    def fit(self):
        return self

    def transform(self):
        return self