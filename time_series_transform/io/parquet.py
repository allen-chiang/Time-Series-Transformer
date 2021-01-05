import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq
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

class Parquet_IO(io_base):
    def __init__(self,dirPaths,time_series,timeSeriesCol,mainCategoryCol,version="1.0"):
        """
        Parquet_IO IO class for apache parquet
        
        Parameters
        ----------
        dirPaths : str
            directory to parquet file
        time_series : Time_Series_Data or Time_Series_Data_Collection
            input data
        timeSeriesCol : str or int
            index of time period column
        mainCategoryCol : str of int
            index of category column
        version : str, optional
            parquet version, by default "1.0"
        """
        super().__init__(time_series, timeSeriesCol, mainCategoryCol)
        if self.dictList is not None:
            self.dictList = time_series
        self.dirPaths = dirPaths
        self.version = version

    def from_parquet(self,columns,partitioning,filters,filesystem):
        """
        from_parquet transform parquet into Time_Series_Data or Time_Series_Data_Collection
        
        Parameters
        ----------
        columns : list
           apache arrow implmentation
        partitioning : list
            apache arrow implmentation
        filters : str
            apache arrow implmentation
        filesystem : str
            apache arrow implmentation 
        
        Returns
        -------
        Time_Series_Data or Time_Series_Data_Collection
        """
        table = pq.read_table(
            source = self.dirPaths,
            columns = columns,
            partitioning = partitioning,
            filters=filters,
            filesystem =filesystem 
        )
        return from_arrow_table(table,self.timeSeriesCol,self.mainCategoryCol)

    def to_parquet(self,expandCategory,expandTime,preprocessType,seperateLabels,partition_cols,isDataset):
        """
        to_parquet transform Time_Series_Data or Time_Series_Data_Collection
        to parquet
        
        Parameters
        ----------
        expandCategory : bool
            whether to expand category
        expandTime : bool
            whether to expand time
        preprocessType : ['ignore','pad','remove']
            preprocess data time across categories
        seperateLabels : bool
            whether to seperate labels and data
        partition_cols : list
            partition columns
        isDataset : bool
            whether output as parquet dataset
        """
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
                version = self.version
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
            version = self.version
        )
        pq.write_table(
            label_table,
            self.dirPaths[1],
            version = self.version
        )
        return


def from_parquet(dirPath, timeSeriesCol, mainCategoryCol,columns=None,partitioning='hive',filters=None,filesystem=None):
    """
    from_parquet transform parquet into Time_Series_Data or Time_Series_Data_Collection
    
    Parameters
    ----------
    dirPaths : str
        directory to parquet file
    time_series : Time_Series_Data or Time_Series_Data_Collection
        input data
    timeSeriesCol : str or int
        index of time period column
    mainCategoryCol : str of int
        index of category column
    columns : list, optional
        columns to fetch, by default None
    partitioning : str, optional
        partition type, by default 'hive'
    filters : str, optional
        parquet filter, by default None
    filesystem : str, optional
        filesystem, by default None
    
    Returns
    -------
    Time_Series_Data or Time_Series_Data_Collection
    """
    pio = Parquet_IO(dirPath,None, timeSeriesCol, mainCategoryCol)
    return pio.from_parquet(columns,partitioning,filters,filesystem)

def to_parquet(dirPaths,time_series_data,expandCategory,expandTime,preprocessType,seperateLabels = False,version='1.0',isDataset=False,partition_cols=None):
    """
    to_parquet transform Time_Series_Data or Time_Series_Data_Collection
        to parquet
    
    Parameters
    ----------
    dirPaths : str
        directory to parquet file
    time_series_data : Time_Series_Data or Time_Series_Data_Collection
        input data
    timeSeriesCol : str or int
        index of time period column
    mainCategoryCol : str of int
        index of category column
    preprocessType : ['ignore','pad','remove']
        preprocess data time across categories
    seperateLabels : bool
        whether to seperate labels and data
    version : str, optional
        parquet version, by default '1.0'
    isDataset : bool, optional
        whether to output as dataset, by default False
    partition_cols : list, optional
        partition columns, by default None
    """
    pio = Parquet_IO(dirPaths,time_series_data,None,None,version)
    return pio.to_parquet(
        expandCategory=expandCategory,
        expandTime=expandTime,
        preprocessType=preprocessType,
        seperateLabels=seperateLabels,
        partition_cols=partition_cols,
        isDataset=isDataset
        )





__all__ = [
    'from_parquet',
    'to_parquet'
]