import pandas as pd
from time_series_transform.transform_core_api.base import Time_Series_Data, Time_Series_Data_Collection
from time_series_transform.io.base import io_base


class Pandas_IO (io_base):
    def __init__(self, time_series, timeSeriesCol, mainCategoryCol):
        super().__init__(time_series, timeSeriesCol, mainCategoryCol)
        if self.dictList is not None:
            self.dictList = time_series
    
    def from_pandas(self):
        if not isinstance(self.dictList,pd.DataFrame):
            raise ValueError("input data must be pandas frame")
        data = self.dictList.to_dict('list')
        if self.mainCategoryCol is None:
            return self.to_single()
        return self.to_collection()
        

    def to_pandas(self,expandTime,expandCategory,preprocessType):
        if isinstance(self.time_series,Time_Series_Data):
            data = self.from_single(expandTime)
            return pd.DataFrame(data)
        if isinstance(self.time_series,Time_Series_Data_Collection):
            data = self.from_collection(expandCategory,expandTime,preprocessType)
            return pd.DataFrame(data)
        raise ValueError("Invalid data type")


def from_pandas(pandasFrame,timeSeriesCol,mainCategoryCol=None):
    pio = Pandas_IO(pandasFrame,timeSeriesCol,mainCategoryCol)
    return pio.from_pandas()

def to_pandas(time_series_data,expandCategory,expandTime,preprocessType):
    if isinstance(time_series_data,Time_Series_Data):
        pio = Pandas_IO(time_series_data,time_series_data.time_seriesIx,None)
        return pio.to_pandas(expandTime,None,None)
    if isinstance(time_series_data,Time_Series_Data_Collection):
        pio = Pandas_IO(
            time_series_data,
            time_series_data._time_series_Ix,
            time_series_data._categoryIx
            )
        return pio.to_pandas(expandTime,expandCategory,preprocessType)
    

__all__ = [
    'from_pandas',
    'to_pandas'
]