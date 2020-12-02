import scipy
import numpy as np
import pandas as pd
import pandas_ta as ta
from time_series_transform.io import *
from time_series_transform.io.pandas import to_pandas
from time_series_transform.transform_core_api.util import *
from time_series_transform.transform_core_api.base import *

class Stock(Time_Series_Data):
    def __init__(self,data,time_index,symbol,High='High',Low='Low',Close='Close',Open='Open',Volume='Volume'):
        super().__init__(data,time_index)
        self.ohlcva ={
            'high':High,
            'close':Close,
            'open':Open,
            'volume':Volume,
            'low':Low,
            'Date':time_index
        }

    def get_technical_indicator(self,strategy):
        dct = {}
        all_info=self._get_all_info()
        for i in self.ohlcva:
            dct[i] = all_info[self.ohlcva[i]]
        df = pd.DataFrame(dct)
        df.ta.strategy(strategy)
        keys = list(self._get_all_info().keys())
        for i in df.columns:
            if i in keys:
                continue
            self.set_data(df[i].values,i)
        return self
    
    @classmethod
    def from_time_series_data(cls,time_series_data,symbol,High='High',Low='Low',Close='Close',Open='Open',Volume='Volume'):
        ohlcva ={
            'high':High,
            'close':Close,
            'open':Open,
            'volume':Volume,
            'low':Low,
            'Date':self.time_index
        }

class Portfolio(Time_Series_Data_Collection):
    def __init__(self,time_series_data,time_index,symbolIx,High='High',Low='Low',Close='Close',Open='Open',Volume='Volume'):
        super().__init__(time_series_data,time_index,symbolIx)
        self.ohlcva ={
            'high':High,
            'close':Close,
            'open':Open,
            'volume':Volume,
            'low':Low,
            'Date':time_index
        }
    
    def get_technial_indicator(self,strategy):
        pass