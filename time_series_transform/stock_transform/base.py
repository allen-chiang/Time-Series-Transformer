import scipy
import numpy as np
import pandas as pd
import pandas_ta as ta
from joblib import Parallel, delayed
from time_series_transform.io.base import io_base
from time_series_transform.io.pandas import to_pandas
from time_series_transform.transform_core_api.util import *
from time_series_transform.transform_core_api.base import *

class Stock(Time_Series_Data):
    def __init__(self,data,time_index,symbol='Default',High='High',Low='Low',Close='Close',Open='Open',Volume='Volume'):
        super().__init__(data,time_index)
        self.ohlcva ={
            'high':High,
            'close':Close,
            'open':Open,
            'volume':Volume,
            'low':Low,
            'Date':time_index
        }
        self.symbol = symbol

    def get_technical_indicator(self,strategy):
        dct = {}
        all_info=self._get_all_info()
        for i in self.ohlcva:
            dct[i] = all_info[self.ohlcva[i]]
        df = pd.DataFrame(dct)
        df.ta.strategy(strategy)
        keys = list(map(lambda x: x.lower(),list(self._get_all_info().keys())))
        for i in df.columns:
            if i in keys:
                continue
            self.set_data(df[i].values,i)
        return self
    
    @classmethod
    def from_time_series_data(cls,time_series_data,High='High',Low='Low',Close='Close',Open='Open',Volume='Volume'):
        ohlcva ={
            'High':High,
            'Close':Close,
            'Open':Open,
            'Volume':Volume,
            'Low':Low
        }
        return cls(
            time_series_data[:],
            time_series_data.time_seriesIx,
            **ohlcva
            )

class Portfolio(Time_Series_Data_Collection):
    def __init__(self,time_series_data,time_index,symbolIx,High='High',Low='Low',Close='Close',Open='Open',Volume='Volume'):
        super().__init__(time_series_data,time_index,symbolIx)
        self.ohlcva ={
            'High':High,
            'Close':Close,
            'Open':Open,
            'Volume':Volume,
            'Low':Low
        }
        self._time_series_data_collection = self._cast_stock_collection()
        

    def _cast_stock_collection(self):
        stock_collection = {}
        for i in self.time_series_data_collection:
            stock_collection[i] = Stock.from_time_series_data(
                self.time_series_data_collection[i],
                High=self.ohlcva['High'],
                Close=self.ohlcva['Close'],
                Open=self.ohlcva['Open'],
                Volume=self.ohlcva['Volume'],
                Low=self.ohlcva['Low'],
                )
        return stock_collection

    def _get_techinal_indicator(self,category,time_series_data,strategy,*args,**kwargs):
        return {category:time_series_data.get_technical_indicator(strategy)}


    def get_technical_indicator(self,strategy,n_jobs =1,verbose = 0,backend='loky',*args,**kwargs):
        dctList = Parallel(n_jobs = n_jobs,backend=backend,verbose=verbose)(delayed(self._get_techinal_indicator)(
            c, 
            self._time_series_data_collection[c],
            strategy,*args,**kwargs) for c in self.time_series_data_collection
        )
        results = {}
        for i in dctList:
            results.update(i)
        self._time_series_data_collection = results
        return self

    @classmethod
    def from_time_series_collection(cls,time_series_data_collection,High='High',Low='Low',Close='Close',Open='Open',Volume='Volume'):
        iobase = io_base(
            time_series_data_collection,
            time_series_data_collection._time_series_Ix,
            time_series_data_collection._categoryIx
            )
        
        return cls(
            time_series_data= Time_Series_Data(iobase.from_collection(False,False,'ignore'),time_series_data_collection._time_series_Ix),
            time_index = time_series_data_collection._time_series_Ix,
            symbolIx= time_series_data_collection._categoryIx,
            High= High,
            Low = Low,
            Close = Close,
            Open = Open,
            Volume = Volume
        )


