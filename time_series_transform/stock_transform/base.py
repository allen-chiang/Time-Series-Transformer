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
    def __init__(self,data,time_index,symbol=None,High='High',Low='Low',Close='Close',Open='Open',Volume='Volume'):
        """
        Stock Basic data structure which inherite from Time_Series_Data.
        
        
        this data structure extend Time_Series_Data and implment Open, Close, High, Low, Volume attributes.
        Also, it has pandas-ta library extension to support making different technical indicator.
        
        Parameters
        ----------
        data : dict of list, optional
            the data of input values; it can have time_index. if it has time_index, the name should
            be passed to time_index parameter, by default None
        time_index : dict of list or string or numeric type, optional
            if it is dict of list the time_series_IX will be initiated by the value.
            else it will use the information and search from data parameter., by default None
        symbol : str, option
            ticker name, by default None
        High : str or int, optional
            the index or name for High, by default 'High'
        Low : str or int, optional
            the index or name for Low, by default 'Low'
        Close : str or int, optional
            the index or name for Close, by default 'Close'
        Open : str or int, optional
            the index or name for Open, by default 'Open'
        Volume : str or int, optional
            the index or name for Volume, by default 'Volume'
        """
        super().__init__(data,time_index)
        self.ohlcva ={
            'High':High,
            'Close':Close,
            'Open':Open,
            'Volume':Volume,
            'Low':Low,
            'Date':time_index
        }
        self.symbol = symbol

    def get_technical_indicator(self,strategy):
        """
        get_technical_indicator making different technical indicator
        
        pandas-ta implmentation
        https://github.com/twopirllc/pandas-ta
        
        Parameters
        ----------
        strategy : Strategy
            pandas-ta strategy
        
        Returns
        -------
        self
        """
        dct = {}
        all_info=self._get_all_info()
        for i in self.ohlcva:
            dct[i] = all_info[self.ohlcva[i]]
        df = pd.DataFrame(dct)
        df.ta.strategy(strategy)
        keys = list(map(lambda x: x.lower(),list(self._get_all_info().keys())))
        for i in self.ohlcva:
            keys.append(str.lower(i))
        for i in df.columns:
            if i in keys:
                continue
            self.set_data(df[i].values,i)
        return self
    
    @classmethod
    def from_time_series_data(cls,time_series_data,symbol=None,High='High',Low='Low',Close='Close',Open='Open',Volume='Volume'):
        """
        from_time_series_data 
        making Stock object from Time_Series_Data class
        
        Parameters
        ----------
        time_series_data : Time_Series_Data
            input Data
        symbol : str, option
            ticker name, by default None
        High : str or int, optional
            the index or name for High, by default 'High'
        Low : str or int, optional
            the index or name for Low, by default 'Low'
        Close : str or int, optional
            the index or name for Close, by default 'Close'
        Open : str or int, optional
            the index or name for Open, by default 'Open'
        Volume : str or int, optional
            the index or name for Volume, by default 'Volume'
        
        Returns
        -------
        Stock
        """
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
            symbol = symbol,
            **ohlcva
            )

class Portfolio(Time_Series_Data_Collection):
    def __init__(self,time_series_data,time_index,symbolIx,High='High',Low='Low',Close='Close',Open='Open',Volume='Volume'):
        """
        Portfolio [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        time_series_data : dict of Time_Series_Data or Time_Series_Data
            if this parameter is a dict of Time_Series_Data, it will directly cast into this class.
            else, it will seperate teh Time_Series_Data according to the categoryIX column.
        time_index : str
            the name of time_seriesIx
        symbolIx : str or int
            the symbol column index of the data
        High : str or int, optional
            the index or name for High, by default 'High'
        Low : str or int, optional
            the index or name for Low, by default 'Low'
        Close : str or int, optional
            the index or name for Close, by default 'Close'
        Open : str or int, optional
            the index or name for Open, by default 'Open'
        Volume : str or int, optional
            the index or name for Volume, by default 'Volume'
        """
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
                symbol= i,
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
        """
        get_technical_indicator making different technical indicator
        
        pandas-ta implmentation
        https://github.com/twopirllc/pandas-ta
        
        
        Parameters
        ----------
        strategy : Strategy
            pandas-ta strategy
        n_jobs : int, optional
            number of processes (joblib), by default 1
        verbose : int, optional
            log level (joblib), by default 0
        backend : str, optional
            backend type (joblib), by default 'loky'
        
        Returns
        -------
        self
        """
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
        """
        from_time_series_collection making Portfolio object from Time_Series_Data_Collection
        
        Parameters
        ----------
        time_series_data_collection : Time_Series_Data_Collection
            input data
        High : str or int, optional
            the index or name for High, by default 'High'
        Low : str or int, optional
            the index or name for Low, by default 'Low'
        Close : str or int, optional
            the index or name for Close, by default 'Close'
        Open : str or int, optional
            the index or name for Open, by default 'Open'
        Volume : str or int, optional
            the index or name for Volume, by default 'Volume'
        
        Returns
        -------
        Portfolio
        """
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


