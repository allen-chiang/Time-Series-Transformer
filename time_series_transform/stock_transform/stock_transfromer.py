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
        return cls(data,'Date',None)

    @classmethod
    def from_stock_engine_date(cls,symbols,start_date,end_date,engine,n_threads=8,*args,**kwargs):
        if isinstance(symbols,list):
            se = Portfolio_Extractor(symbols,engine,*args,**kwargs)
            data = se.get_date(start_date,end_date,n_threads = n_threads)
            return cls(data,'Date','symbol')
        se = Stock_Extractor(symbols,engine,*args,**kwargs)
        data = se.get_date(start_date,end_date)
        return cls(data,'Date',None)

    @classmethod
    def from_pandas(
                cls, 
                pandasFrame,
                timeSeriesCol,
                mainCategoryCol,
                High='High',
                Low='Low',
                Close='Close',
                Open='Open',
                Volume='Volume'):
        data = io.from_pandas(pandasFrame,timeSeriesCol,mainCategoryCol)
        data = _time_series_data_to_stock_data(data,mainCategoryCol,High,Low,Close,Open,Volume)
        return cls(data,timeSeriesCol,mainCategoryCol)

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
                Volume):
        data = io.from_numpy(numpyData,timeSeriesCol,mainCategoryCol)
        data = _time_series_data_to_stock_data(data,mainCategoryCol,High,Low,Close,Open,Volume)
        return cls(data,timeSeriesCol,mainCategoryCol)


    @classmethod
    def from_time_series_transformer(
                cls,
                time_series_transformer,
                High='High',
                Low='Low',
                Close='Close',
                Open='Open',
                Volume='Volume'):
        data = time_series_transformer.time_series_data
        timeCol = time_series_transformer.timeSeriesCol
        symbolIx = time_series_transformer.mainCategoryCol
        return cls(data,timeCol,symbolIx,High,Low,Close,Open,Volume)
        

    @classmethod
    def from_feather(
                cls,
                feather_dir,
                timeSeriesCol,
                symbolIx,
                columns=None,
                High='High',
                Low='Low',
                Close='Close',
                Open='Open',
                Volume='Volume'):
        data = io.from_feather(
            feather_dir,
            timeSeriesCol,
            symbolIx,
            columns
            )
        return cls(data,timeSeriesCol,symbolIx,High,Low,Close,Open,Volume)
    
    @classmethod
    def from_parquet(
                cls,
                parquet_dir,
                timeSeriesCol,
                symbolIx,
                columns = None,
                partitioning='hive',
                filters=None,
                filesystem=None,
                High='High',
                Low='Low',
                Close='Close',
                Open='Open',
                Volume='Volume'):
        data = io.from_parquet(
            parquet_dir,
            timeSeriesCol,
            symbolIx,
            columns,
            partitioning,
            filters,
            filesystem
            )
        return cls(data,timeSeriesCol,symbolIx,High,Low,Close,Open,Volume)
    
    @classmethod
    def from_arrow_table(
                cls,
                arrow_table,
                timeSeriesCol,
                symbolIx,
                High='High',
                Low='Low',
                Close='Close',
                Open='Open',
                Volume='Volume'):
        data = io.from_arrow_table(arrow_table,timeSeriesCol,symbolIx)
        return cls(data,timeSeriesCol,symbolIx,High,Low,Close,Open,Volume)


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