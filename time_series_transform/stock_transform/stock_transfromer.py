import collections
from time_series_transform import io
from time_series_transform.stock_transform.base import Stock, Portfolio
from time_series_transform.transform_core_api.time_series_transformer import Time_Series_Transformer
from time_series_transform.transform_core_api.base import Time_Series_Data,Time_Series_Data_Collection
from time_series_transform.stock_transform.stock_extractor import Stock_Extractor, Portfolio_Extractor
from time_series_transform.plot import *

class Stock_Transformer(Time_Series_Transformer):
    def __init__(self,time_series_data,time_seriesIx,symbolIx,symbolName=None,High='High',Low='Low',Close='Close',Open='Open',Volume='Volume'):
        """
        Stock_Transformer the class for Stock and Portfolio data manipulation
        
        This class inhereite from Time_Series_Transform.
        it can perform different data manipulation: making lag data,
        lead data, lag sequence data, making technical indicator through pandas-ta api, 
        or do a customize data manipulation.
        It also built in native plot and io functions. IO function currently
        support pandas DataFrame, numpy ndArray, apache arrow table , apache feather,
        and apache parquet. Besides these io, it currently support fetching data from
        yahoo finance and investment.com data through yfinance and investpy api.

        yfinance: https://github.com/ranaroussi/yfinance
        investpy: https://github.com/alvarobartt/investpy
        
        Parameters
        ----------
        time_series_data : dict of list, Stock, or Portfolio
            the value of data.
        time_seriesIx : str
            the name of time_seriesIx
        symbolIx : str or int
            the symbol column index of the data
        symbolName : str, optional
            tiker name used only when there is single stock, by default None
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
        super().__init__(time_series_data,time_seriesIx, symbolIx)
        if not isinstance(time_series_data, (Stock,Portfolio)):
            self.time_series_data =_time_series_data_to_stock_data(self.time_series_data,symbolName,High,Low,Close,Open,Volume)
        self.plot = StockPlot(self.time_series_data)

    @classmethod
    def from_stock_engine_period(cls,symbols,period,engine,n_threads=8,*args,**kwargs):
        """
        from_stock_engine_period fetching data from online
        
        the current engine support yfinance and investpy
        
        Parameters
        ----------
        symbols : str or list
            ticker name
        period : str
            period of the data
            for example, 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max 
        engine : ['yahoo','investing']
            fetching api
        n_threads : int, optional
            multi-thread fetching support only when symbols is a list, by default 8
        
        Returns
        -------
        Stock_Transformer
        """
        if isinstance(symbols,list):
            se = Portfolio_Extractor(symbols,engine,*args,**kwargs)
            data = se.get_period(period,n_threads = n_threads)
            return cls(data,'Date','symbol')
        se = Stock_Extractor(symbols,engine,*args,**kwargs)
        data = se.get_period(period)
        return cls(data,'Date',None,symbols)

    @classmethod
    def from_stock_engine_date(cls,symbols,start_date,end_date,engine,n_threads=8,*args,**kwargs):
        """
        from_stock_engine_date [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        symbols : str or list
            ticker name
        start_date : str
            start of the data
            format: "%Y-%m-%d", eg "2020-02-20"
        end_date : str
            end of the data
            format: "%Y-%m-%d", eg "2020-02-20"
        engine : ['yahoo','investing']
            fetching api
        n_threads : int, optional
            multi-thread fetching support only when symbols is a list, by default 8
        
        
        Returns
        -------
        Stock_Transformer
        """
        if isinstance(symbols,list):
            se = Portfolio_Extractor(symbols,engine,*args,**kwargs)
            data = se.get_date(start_date,end_date,n_threads = n_threads)
            return cls(data,'Date','symbol')
        se = Stock_Extractor(symbols,engine,*args,**kwargs)
        data = se.get_date(start_date,end_date)
        return cls(data,'Date',None,symbols)

    @classmethod
    def from_stock_engine_intraday(cls,symbols,start_date,end_date,engine='yahoo', interval = '1m',n_threads=8,*args,**kwargs):
        if isinstance(symbols,list):
            se = Portfolio_Extractor(symbols,engine,*args,**kwargs)
            data = se.get_intra_day(start_date,end_date, interval=interval,n_threads = n_threads)
            return cls(data,'Datetime','symbol')
        se = Stock_Extractor(symbols,engine,*args,**kwargs)
        data = se.get_intra_day(start_date,end_date, interval=interval)
        return cls(data,'Datetime',None,symbols)

    @classmethod
    def from_pandas(
                cls, 
                pandasFrame,
                timeSeriesCol,
                mainCategoryCol,
                symbolName = None,
                High='High',
                Low='Low',
                Close='Close',
                Open='Open',
                Volume='Volume'):
        """
        from_pandas
        
        from_pandas import data from pandas dataFrame
        
        Parameters
        ----------
        pandasFrame : pandas DataFrame
            input data
        timeSeriesCol : str or numeric
            time series column name
        mainCategoryCol : str or numeric
            main category name
        symbolName : str or numeric, option
            ticker name only used when single stock, by default None
        High : str or int, optional
            the column name for High, by default 'High'
        Low : str or int, optional
            the column name for Low, by default 'Low'
        Close : str or int, optional
            the column name for Close, by default 'Close'
        Open : str or int, optional
            the column name for Open, by default 'Open'
        Volume : str or int, optional
            the column name for Volume, by default 'Volume'
        
        Returns
        -------
        Stock_Transformer
        """
        data = io.from_pandas(pandasFrame,timeSeriesCol,mainCategoryCol)
        data = _time_series_data_to_stock_data(data,mainCategoryCol,High,Low,Close,Open,Volume)
        return cls(data,timeSeriesCol,mainCategoryCol,symbolName)

    @classmethod
    def from_numpy(
                cls,
                numpyData,
                timeSeriesCol,
                mainCategoryCol,
                High,
                Low,
                Close,
                Open,
                Volume,
                symbolName = None):
        """
        from_numpy from_numpy import data from numpy
        
        Parameters
        ----------
        numpyData : numpy ndArray
            input data
        timeSeriesCol : int
            index of time series column
        mainCategoryCol : int
            index of main category column
        High : int, optional
            the column index for High, by default 'High'
        Low : int, optional
            the column index for Low, by default 'Low'
        Close : int, optional
            the column index for Close, by default 'Close'
        Open : int, optional
            the column index for Open, by default 'Open'
        Volume : int, optional
            the column index for Volume, by default 'Volume'
        symbolName : str or numeric, option
            ticker name only used when single stock, by default None
        
        Returns
        -------
        Stock_Transformer
        """
        data = io.from_numpy(numpyData,timeSeriesCol,mainCategoryCol)
        data = _time_series_data_to_stock_data(data,symbolName,High,Low,Close,Open,Volume)
        return cls(data,timeSeriesCol,mainCategoryCol,symbolName)


    @classmethod
    def from_time_series_transformer(
                cls,
                time_series_transformer,
                symbolName = None,
                High='High',
                Low='Low',
                Close='Close',
                Open='Open',
                Volume='Volume'):
        """
        from_time_series_transformer making Stock_Transformer from Time_Series_Transformer
        
        Parameters
        ----------
        time_series_transformer : Time_Series_Transformer
            input data
        symbolName : str or numeric, option
            ticker name only used when single stock, by default None
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
        Stock_Transformer
        """
        data = time_series_transformer.time_series_data
        timeCol = time_series_transformer.timeSeriesCol
        symbolIx = time_series_transformer.mainCategoryCol
        return cls(data,timeCol,symbolIx,symbolName,High,Low,Close,Open,Volume)
        

    @classmethod
    def from_feather(
                cls,
                feather_dir,
                timeSeriesCol,
                symbolIx,
                symbolName = None,
                columns=None,
                High='High',
                Low='Low',
                Close='Close',
                Open='Open',
                Volume='Volume'):
        """
        from_feather import data from feather
        
        Parameters
        ----------
        feather_dir : str
            directory of feather file
        timeSeriesCol : str or numeric
            time series column name
        symbolIx : str or numeric
            main category name
        symbolName : str or numeric, option
            ticker name only used when single stock, by default None
        columns : str or numeric, optional
            target columns (apache arrow implmentation), by default None
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
        Stock_Transformer
        """
        data = io.from_feather(
            feather_dir,
            timeSeriesCol,
            symbolIx,
            columns
            )
        return cls(data,timeSeriesCol,symbolIx,symbolName,High,Low,Close,Open,Volume)
    
    @classmethod
    def from_parquet(
                cls,
                parquet_dir,
                timeSeriesCol,
                symbolIx,
                symbolName = None,
                columns = None,
                partitioning='hive',
                filters=None,
                filesystem=None,
                High='High',
                Low='Low',
                Close='Close',
                Open='Open',
                Volume='Volume'):
        """
        from_parquet import data from parquet file
        
        Parameters
        ----------
        parquet_dir : str
            directory of parquet file
        timeSeriesCol : str or numeric
            time series column name
        symbolIx : str or numeric
            main category name
        symbolName : str or numeric, option
            ticker name only used when single stock, by default None
        columns : str or numeric, optional
            target columns (apache arrow implmentation), by default None
        partitioning : str, optional
            type of partitioning, by default 'hive'
        filters : str, optional
            filter (apache arrow implmentation), by default None
        filesystem : str, optional
            filesystem (apache arrow implmentation), by default None
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
        Stock_Transformer
        """
        data = io.from_parquet(
            parquet_dir,
            timeSeriesCol,
            symbolIx,
            columns,
            partitioning,
            filters,
            filesystem
            )
        return cls(data,timeSeriesCol,symbolIx,symbolName,High,Low,Close,Open,Volume)
    
    @classmethod
    def from_arrow_table(
                cls,
                arrow_table,
                timeSeriesCol,
                symbolIx,
                symbolName = None,
                High='High',
                Low='Low',
                Close='Close',
                Open='Open',
                Volume='Volume'):
        """
        from_arrow_table [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        arrow_table : arrow table
            input data
        timeSeriesCol : str or numeric
            time series column name
        symbolIx : str or numeric
            main category name
        symbolName : str or numeric, option
            ticker name only used when single stock, by default None
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
        Stock_Transformer
        """
        data = io.from_arrow_table(arrow_table,timeSeriesCol,symbolIx)
        return cls(data,timeSeriesCol,symbolIx,symbolName,High,Low,Close,Open,Volume)


    def get_technial_indicator(self,strategy,n_jobs=1,verbose=10,backend='loky'):
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
        if isinstance(self.time_series_data,Portfolio):
            self.time_series_data= self.time_series_data.get_technical_indicator(
                strategy,
                n_jobs=n_jobs,
                verbose = verbose,
                backend = backend
                )
            return self
        self.time_series_data= self.time_series_data.get_technical_indicator(
            strategy
            )
        return self


def _time_series_data_to_stock_data(time_series_data,symbolName,High,Low,Close,Open,Volume):
    res = None
    if isinstance(time_series_data,Time_Series_Data):
        res = Stock.from_time_series_data(
            time_series_data= time_series_data,
            symbol = symbolName,
            High = High,
            Low = Low,
            Close = Close,
            Open = Open,
            Volume = Volume
            )
    else:
        res = Portfolio.from_time_series_collection(
            time_series_data_collection= time_series_data,
            High = High,
            Low = Low,
            Close = Close,
            Open = Open,
            Volume = Volume
        )
    return res