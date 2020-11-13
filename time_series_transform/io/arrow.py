import numpy as np
import pandas as pd
import parrow as pa
from parrow.parquet as pq
from time_series_transform.io.base import io_base
from time_series_transform.transform_core_api.base import (
    Time_Series_Data, 
    Time_Series_Data_Collection
    )

class Arrow_IO(io_base):
    def __init__(self,dirPath,time_series,timeSeriesCol,mainCategoryCol):
        super().__init__(time_series, timeSeriesCol, mainCategoryCol)
        if self.dictList is not None:
            self.dictList = time_series
        self.dirPath = dirPath

    def from_arrow_table(self):
        pass

    def to_arrow_table(self):
        pass

    def to_arrow_record_batch(self):
        pass

    def from_arrow_record_batch(self):
        pass


def from_arrow_table(dirPath, time_series, timeSeriesCol, mainCategoryCol):
    pass

def to_arrow_table(time_series_data,expandCategory,expandTime,preprocessType,seperateLabels = False):
    pass

def from_arrow_record_batch(dirPath, time_series, timeSeriesCol, mainCategoryCol):
    pass

def to_arrow_record_batch(time_series_data,expandCategory,expandTime,preprocessType,seperateLabels = False):
    pass



__all__ = [
    'from_arrow_table',
    'to_arrow_table'
]