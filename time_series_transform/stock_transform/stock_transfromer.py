import collections
from time_series_transform import io
from time_series_transform.stock_transform.base import Stock, Portfolio
from time_series_transform.transform_core_api.time_series_transformer import Time_Series_Transformer
from time_series_transform.transform_core_api.base import Time_Series_Data,Time_Series_Data_Collection
from time_series_transform.stock_transform.stock_extractor import Stock_Extractor, Portfolio_Extractor


class Stock_Transformer(Time_Series_Transformer):
    def __init__(self,time_series_data,time_seriesIx,symbolIx,High='High',Low='Low',Close='Close',Open='Open',Volume='Volume'):
        super().__init__(time_series_data,time_seriesIx, symbolIx)
        if not isinstance(time_series_data, (Stock,Portfolio)):
            self.time_series_data =_time_series_data_to_stock_data(self.time_series_data,self.mainCategoryCol,High,Low,Close,Open,Volume)

    @classmethod
    def from_stock_engine_period(cls,symbols,period,engine,n_threads=8,*args,**kwargs):
        if isinstance(symbols,list):
            se = Portfolio_Extractor(symbols,engine,*args,**kwargs)
            data = se.get_period(period,n_threads = n_threads)
            return cls(data,'Date','symbol')
        se = Stock_Extractor(symbols,engine,*args,**kwargs)
        data = se.get_period(period)
        return cls(data,'Date','symbol')

    @classmethod
    def from_stock_engine_date(cls,symbols,start_date,end_date,engine,n_threads=8,*args,**kwargs):
        if isinstance(symbols,list):
            se = Portfolio_Extractor(symbols,engine,*args,**kwargs)
            data = se.get_date(start_date,end_date,n_threads = n_threads)
            return cls(data,'Date','symbol')
        se = Stock_Extractor(symbols,engine,*args,**kwargs)
        data = se.get_period(start_date,end_date)
        return cls(data,'Date','symbol')

    @classmethod
    def from_pandas(cls, pandasFrame,timeSeriesCol,mainCategoryCol,High='High',Low='Low',Close='Close',Open='Open',Volume='Volume'):
        data = io.from_pandas(pandasFrame,timeSeriesCol,mainCategoryCol)
        data = _time_series_data_to_stock_data(data,mainCategoryCol,High,Low,Close,Open,Volume)
        return cls(data,timeSeriesCol,mainCategoryCol)

    @classmethod
    def from_numpy(cls,numpyData,timeSeriesCol,mainCategoryCol,High,Low,Close,Open,Volume):
        data = io.from_numpy(numpyData,timeSeriesCol,mainCategoryCol)
        data = _time_series_data_to_stock_data(data,mainCategoryCol,High,Low,Close,Open,Volume)
        return cls(data,timeSeriesCol,mainCategoryCol)

    def get_technial_indicator(self,strategy,n_jobs=1,verbose=10,backend='loky'):
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



def _time_series_data_to_stock_data(time_series_data,mainCategoryCol,High,Low,Close,Open,Volume):
    res = None
    if isinstance(time_series_data,Time_Series_Data):
        res = Stock.from_time_series_data(
            time_series_data= time_series_data,
            symbol = time_series_data[:,[mainCategoryCol]][mainCategoryCol][0],
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