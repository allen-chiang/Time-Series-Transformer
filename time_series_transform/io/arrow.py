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
        """
        Arrow_IO IO class for apache arrow
        
        there are two types of transformation:
        apache arrow batch record and apache arrow table
        
        Parameters
        ----------
        time_series : Time_Series_Data or Time_Series_Data_Collection
            input data
        timeSeriesCol : str or int
            index of time period column
        mainCategoryCol : str of int
            index of category column
        """
        super().__init__(time_series, timeSeriesCol, mainCategoryCol)
        if self.dictList is not None:
            self.dictList = time_series

    def from_arrow_table(self):
        """
        from_arrow_table transform arrow table to Time_Series_Data or Time_Series_Data_Collection
        
        Returns
        -------
        Time_Series_Data or Time_Series_Data_Collection
        """
        df = self.dictList.to_pandas()
        return from_pandas(df,self.timeSeriesCol,self.mainCategoryCol)

    def from_arrow_record_batch(self):
        """
        from_arrow_record_batch from_arrow_table transform arrow record batch
         to Time_Series_Data or Time_Series_Data_Collection
        
        Returns
        -------
        Time_Series_Data or Time_Series_Data_Collection
        """
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
        """
        to_arrow_table transform Time_Series_Data or Time_Series_Data_Collection
        to arrow table
        
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
        
        Returns
        -------
        arrow table
        """
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
        """
        to_arrow_record_batch 
        transform Time_Series_Data or Time_Series_Data_Collection
        to arrow record batch
        
        
        Parameters
        ----------
        max_chunksize : int
            max size of record batch
        expandCategory : bool
            whether to expand category
        expandTime : bool
            whether to expand time
        preprocessType : ['ignore','pad','remove']
            preprocess data time across categories
        seperateLabels : bool
            whether to seperate labels and data
        
        Returns
        -------
        arrow record batch
        """
        if seperateLabels == False:
            table = self.to_arrow_table(expandCategory,expandTime,preprocessType,seperateLabels)
            return table.to_batches(max_chunksize)
        table,labelTable = self.to_arrow_table(expandCategory,expandTime,preprocessType,seperateLabels)
        return table.to_batches(max_chunksize),labelTable.to_batches(max_chunksize)


def from_arrow_table(time_series, timeSeriesCol, mainCategoryCol):
    """
    from_arrow_table transform arrow table
         to Time_Series_Data or Time_Series_Data_Collection
    
    Parameters
    ----------
    time_series : Time_Series_Data or Time_Series_Data_Collection
        input data
    timeSeriesCol : str or int
        index of time period column
    mainCategoryCol : str of int
        index of category column
    Returns
    -------
    arrow table
    """
    aio = Arrow_IO(time_series, timeSeriesCol, mainCategoryCol)
    return aio.from_arrow_table()
    
def from_arrow_record_batch(time_series, timeSeriesCol, mainCategoryCol):
    """
    from_arrow_record_batch transform arrow record batch
    to Time_Series_Data or Time_Series_Data_Collection
    
    Parameters
    ----------
    time_series : Time_Series_Data or Time_Series_Data_Collection
        input data
    timeSeriesCol : str or int
        index of time period column
    mainCategoryCol : str of int
        index of category column
    Returns
    -------
    arrow record batch
    """
    aio = Arrow_IO(time_series, timeSeriesCol, mainCategoryCol)
    return aio.from_arrow_record_batch()

def to_arrow_table(time_series,expandCategory,expandTime,preprocessType,seperateLabels = False):
    """
    to_arrow_table Time_Series_Data or Time_Series_Data_Collection
    to arrow table
    
    Parameters
    ----------
    time_series : Time_Series_Data or Time_Series_Data_Collection
        input data
    expandCategory : bool
        whether to expand category
    expandTime : bool
        whether to expand time
    preprocessType : ['ignore','pad','remove']
        preprocess data time across categories
    seperateLabels : bool
        whether to seperate labels and data
    
    Returns
    -------
    arrow table
    """
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
    """
    to_arrow_record_batch [summary]
    
    [extended_summary]
    
    Parameters
    ----------
    time_series : Time_Series_Data or Time_Series_Data_Collection
        input data
    max_chunksize : int
        max size of record batch
    expandCategory : bool
        whether to expand category
    expandTime : bool
        whether to expand time
    preprocessType : ['ignore','pad','remove']
        preprocess data time across categories
    seperateLabels : bool
        whether to seperate labels and data
    
    Returns
    -------
    arrow record batch
    """
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