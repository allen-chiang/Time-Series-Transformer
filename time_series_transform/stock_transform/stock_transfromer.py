import collections
from time_series_transform import io
from time_series_transform.transform_core_api.time_series_transformer import Time_Series_Transformer
from time_series_transform.transform_core_api.base import Time_Series_Data,Time_Series_Data_Collection
from time_series_transform.stock_transform.base import Stock, Portfolio


class Stock_Transformer(Time_Series_Transformer):
    def __init__(self,time_series_data,time_seriesIx,symbolIx,High='High',Low='Low',Close='Close',Open='Open',Volume='Volume'):
        super().__init__(time_series_data,time_seriesIx, symbolIx)
        if not isinstance(time_series_data, (Stock,Portfolio)):
            self.time_series_data =_time_series_data_to_stock_data(self.time_series_data,self.mainCategoryCol,High,Low,Close,Open,Volume)

    @classmethod
    def from_stock_engine_period(cls):
        pass

    @classmethod
    def from_stock_engine_date(cls):
        pass

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