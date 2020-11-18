import numpy as np
import pandas as pd
import parrow as pa
from parrow import parquet as pq
from time_series_transform.io.base import io_base
from time_series_transform.io.pandas import (
    from_pandas,
    to_pandas
)
from time_series_transform.io.arrow import (
    to_arrow_table
)
from time_series_transform.transform_core_api.base import (
    Time_Series_Data, 
    Time_Series_Data_Collection
    )

class Parquet_IO(io_base):
    def __init__(self,dirPaths,time_series,timeSeriesCol,mainCategoryCol,version="1.0"):
        super().__init__(time_series, timeSeriesCol, mainCategoryCol)
        if self.dictList is not None:
            self.dictList = time_series
        self.dirPaths = dirPaths
        self.version = version

    def from_parquet(self):
        pass

    def to_parquet(self,expandCategory,expandTime,preprocessType,seperateLabels,partition_cols,isDataset):
        if seperateLabels ==False:
            table = to_arrow_table(
                time_series = self.time_series,
                expandCategory = expandCategory,
                expandTime= expandTime,
                preprocessType = preprocessType,
                seperateLabels = seperateLabels
                )
            if isDataset:
                pq.write_to_dataset(
                    table,
                    root_path = self.dirPaths,
                    partition_cols = partition_cols,
                    version = self.version
                    )
                return
            pq.write_table(
                table,
                self.dirPaths,
                vresion = self.version
            )
            return
        table, label_table = to_arrow_table(
                time_series = self.time_series,
                expandCategory = expandCategory,
                expandTime= expandTime,
                preprocessType = preprocessType,
                seperateLabels = seperateLabels
                )
        if isDataset:
            pq.write_to_dataset(
                table,
                root_path = self.dirPaths[0],
                partition_cols = partition_cols,
                version = self.version
                )
            pq.write_to_dataset(
                label_table,
                root_path = self.dirPaths[1],
                partition_cols = partition_cols,
                version = self.version
                )
            return
        pq.write_table(
            table,
            self.dirPaths[0],
            vresion = self.version
        )
        pq.write_table(
            label_table,
            self.dirPaths[1],
            vresion = self.version
        )
        return

            


def from_parquet(dirPath, time_series, timeSeriesCol, mainCategoryCol):
    pass

def to_parquet(time_series_data,expandCategory,expandTime,preprocessType,seperateLabels = False):
    pass





__all__ = [
    'from_parquet',
    'to_parquet'
]