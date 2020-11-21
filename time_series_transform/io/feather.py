import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import feather as pf
from time_series_transform.io.base import io_base
from time_series_transform.io.pandas import (
    from_pandas,
    to_pandas
)
from time_series_transform.io.arrow import (
    to_arrow_table,
    from_arrow_table
)
from time_series_transform.transform_core_api.base import (
    Time_Series_Data, 
    Time_Series_Data_Collection
    )

class Feather_IO(io_base):
    def __init__(self,dirPaths,time_series,timeSeriesCol,mainCategoryCol,version=1):
        super().__init__(time_series, timeSeriesCol, mainCategoryCol)
        if self.dictList is not None:
            self.dictList = time_series
        self.dirPaths = dirPaths
        self.version = version

    def from_feather(self,columns):
        table = pf.read_table(
            source= self.dirPaths,
            columns = columns
        )
        return from_arrow_table(table,self.timeSeriesCol,self.mainCategoryCol)

    def to_feather(self,expandCategory,expandTime,preprocessType,seperateLabels,chunksize):
        if seperateLabels ==False:
            table = to_arrow_table(
                time_series = self.time_series,
                expandCategory = expandCategory,
                expandTime= expandTime,
                preprocessType = preprocessType,
                seperateLabels = seperateLabels
                )
            pf.write_feather(table,self.dirPaths,version = self.version,chunksize=chunksize)
        table, label_table = to_arrow_table(
                time_series = self.time_series,
                expandCategory = expandCategory,
                expandTime= expandTime,
                preprocessType = preprocessType,
                seperateLabels = seperateLabels
                )
        pf.write_feather(table,self.dirPaths,version = self.version,chunksize=chunksize)
        pf.write_feather(label_table,self.dirPaths,version = self.version,chunksize=chunksize)



def from_feather(dirPath, timeSeriesCol, mainCategoryCol,columns=None):
    pio = Feather_IO(dirPath,None, timeSeriesCol, mainCategoryCol)
    return pio.from_feather(columns)

def to_feather(dirPaths,time_series_data,expandCategory,expandTime,preprocessType,seperateLabels = False,version=1,chunksize = None):
    pio = Feather_IO(dirPaths,time_series_data,None,None,version)
    return pio.to_feather(
        expandCategory=expandCategory,
        expandTime=expandTime,
        preprocessType=preprocessType,
        seperateLabels=seperateLabels,
        chunksize=chunksize
        )





__all__ = [
    'from_feather',
    'to_feather'
]