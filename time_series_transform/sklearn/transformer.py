import numpy as np
import pandas as pd
from sklearn.base import (BaseEstimator, TransformerMixin)



class Base_Time_Series_Transformer(self,BaseEstimator, TransformerMixin):

    def __init__(self,time_col,category_col=None,len_preprocessing = 'ignore'):
        self._time_col = time_col 
        self._category_col = category_col
        self._time_series_cache = None
        self._len_preprocessing = len_preprocessing
 
    def fit( self, X, y = None ):
        return self 

    def transform( self, X, y = None ):
        return X[ self._feature_names ] 

    def get_time_series_cache (self):
        return self._time_series_cache


class Lag_Transformer(Base_Time_Series_Transformer):
    def __init__(self,time_col,category_col=None,len_preprocessing = 'ignore'):
        super().__init__(time_col,category_col,len_preprocessing)

    def fit(self):
        return self

    def transform(self):
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