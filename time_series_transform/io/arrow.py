import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq
from time_series_transform.io.base import io_base
from time_series_transform.io.pandas import (
    from_pandas,
    to_pandas
)
from time_series_transform.transform_core_api.base import (
    Time_Series_Data, 
    Time_Series_Data_Collection
    )

class Arrow_IO(io_base):
    def __init__(self,time_series,timeSeriesCol,mainCategoryCol):
        super().__init__(time_series, timeSeriesCol, mainCategoryCol)
        if self.dictList is not None:
            self.dictList = time_series

    def from_arrow_table(self):
        df = self.dictList.to_pandas()
        return from_pandas(df,self.timeSeriesCol,self.mainCategoryCol)

    def from_arrow_record_batch(self):
        df = None
        if isinstance(self.dictList,list):
            for ix,v in enumerate(self.dictList):
                if ix == 0:
                    df = v.to_pandas()
                    continue
                df = df.append(v.to_pandas(),ignore_index = True)
            return from_pandas(df,self.timeSeriesCol,self.mainCategoryCol)
        return from_pandas(self.dictList.to_pandas(),self.timeSeriesCol,self.mainCategoryCol)

    def to_arrow_table(self,expandCategory,expandTime,preprocessType,seperateLabels):
        if seperateLabels == False:
            df = to_pandas(
                time_series_data = self.time_series,
                expandCategory = expandCategory,
                expandTime = expandTime,
                preprocessType= preprocessType,
                seperateLabels= seperateLabels
                )
            return pa.Table.from_pandas(df,preserve_index = False)
        
        df,labelDf = to_pandas(
                time_series_data = self.time_series,
                expandCategory = expandCategory,
                expandTime = expandTime,
                preprocessType= preprocessType,
                seperateLabels= seperateLabels
                )
        return pa.Table.from_pandas(df,preserve_index = False),pa.Table.from_pandas(labelDf,preserve_index = False)

    def to_arrow_record_batch(self,max_chunksize,expandCategory,expandTime,preprocessType,seperateLabels):
        if seperateLabels == False:
            table = self.to_arrow_table(expandCategory,expandTime,preprocessType,seperateLabels)
            return table.to_batches(max_chunksize)
        table,labelTable = self.to_arrow_table(expandCategory,expandTime,preprocessType,seperateLabels)
        return table.to_batches(max_chunksize),labelTable.to_batches(max_chunksize)


def from_arrow_table(time_series, timeSeriesCol, mainCategoryCol):
    aio = Arrow_IO(time_series, timeSeriesCol, mainCategoryCol)
    return aio.from_arrow_table()
    
def from_arrow_record_batch(time_series, timeSeriesCol, mainCategoryCol):
    aio = Arrow_IO(time_series, timeSeriesCol, mainCategoryCol)
    return aio.from_arrow_record_batch()

def to_arrow_table(time_series,expandCategory,expandTime,preprocessType,seperateLabels = False):
    aio = Arrow_IO(time_series,
                    None,
                    None
                   )
    return aio.to_arrow_table(
        expandCategory= expandCategory,
        expandTime=expandTime,
        preprocessType=preprocessType,
        seperateLabels = seperateLabels
        )


def to_arrow_record_batch(time_series,max_chunksize,expandCategory,expandTime,preprocessType,seperateLabels = False):
    aio = Arrow_IO(time_series,
                   None,
                   None
                   )
    return aio.to_arrow_record_batch(
        max_chunksize =max_chunksize,
        expandCategory= expandCategory,
        expandTime=expandTime,
        preprocessType=preprocessType,
        seperateLabels = seperateLabels
        )



__all__ = [
    'from_arrow_table',
    'to_arrow_table',
    'to_arrow_record_batch',
    'from_arrow_record_batch'
]