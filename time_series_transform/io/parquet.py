import numpy as np
import pandas as pd
import parrow as pa
from parrow.parquet as pq
from time_series_transform.io.base import io_base
from time_series_transform.transform_core_api.base import (
    Time_Series_Data, 
    Time_Series_Data_Collection
    )

class Parquet_IO(io_base):
    def __init__(self,dirPath,time_series,timeSeriesCol,mainCategoryCol):
        super().__init__(time_series, timeSeriesCol, mainCategoryCol)
        if self.dictList is not None:
            self.dictList = time_series
        self.dirPath = dirPath

    def from_parquet(self):
        pass

    def to_parquet(self):
        pass


def from_parquet(dirPath, time_series, timeSeriesCol, mainCategoryCol):
    pass

def to_parquet(time_series_data,expandCategory,expandTime,preprocessType,seperateLabels = False):
    pass


__all__ = [
    'from_parquet',
    'to_parquet'
]