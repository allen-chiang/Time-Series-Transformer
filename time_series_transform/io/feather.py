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
        """
        Feather_IO class for apache feather
      
        Parameters
        ----------
        dirPaths : str
            directory to feather file
        time_series : Time_Series_Data or Time_Series_Data_Collection
            input data
        timeSeriesCol : str or int
            index of time period column
        mainCategoryCol : str of int
            index of category column
        version : int, optional
            feather version, by default 1
        """
        super().__init__(time_series, timeSeriesCol, mainCategoryCol)
        if self.dictList is not None:
            self.dictList = time_series
        self.dirPaths = dirPaths
        self.version = version

    def from_feather(self,columns):
        """
        from_feather transform feather to Time_Series_Data or Time_Series_Collection
        
        Parameters
        ----------
        columns : list of str
            column names to fetch
        
        Returns
        -------
        Time_Series_Data or Time_Series_Collection
        """
        table = pf.read_table(
            source= self.dirPaths,
            columns = columns
        )
        return from_arrow_table(table,self.timeSeriesCol,self.mainCategoryCol)

    def to_feather(self,expandCategory,expandTime,preprocessType,seperateLabels,chunksize):
        """
        to_feather transform Time_Series_Data or Time_Series_Data_Collection
        to feather file
        
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
        chunksize : int
            size of feather file
        """
        if seperateLabels ==False:
            table = to_arrow_table(
                time_series = self.time_series,
                expandCategory = expandCategory,
                expandTime= expandTime,
                preprocessType = preprocessType,
                seperateLabels = seperateLabels
                )
            pf.write_feather(table,self.dirPaths,version = self.version,chunksize=chunksize)
            return
        table, label_table = to_arrow_table(
                time_series = self.time_series,
                expandCategory = expandCategory,
                expandTime= expandTime,
                preprocessType = preprocessType,
                seperateLabels = seperateLabels
                )
        pf.write_feather(table,self.dirPaths[0],version = self.version,chunksize=chunksize)
        pf.write_feather(label_table,self.dirPaths[1],version = self.version,chunksize=chunksize)



def from_feather(dirPath, timeSeriesCol, mainCategoryCol,columns=None):
    """
    from_feather read feather file into Time_Series_Data or Time_Series_Data_Collection
    
    Parameters
    ----------
    dirPaths : str
        directory to feather file
    timeSeriesCol : str or int
        index of time period column
    mainCategoryCol : str of int
        index of category column
        columns : list of str
            column names to fetch
    
    Returns
    -------
    Time_Series_Data or Time_Series_Collection
    """
    pio = Feather_IO(dirPath,None, timeSeriesCol, mainCategoryCol)
    return pio.from_feather(columns)

def to_feather(dirPaths,time_series_data,expandCategory,expandTime,preprocessType,seperateLabels = False,version=1,chunksize = None):
    """
    to_feather 
    transform Time_Series_Data or Time_Series_Data_Collection
    to feather file
    
    Parameters
    ----------
    dirPaths : str
        directory to feather file
    time_series_data : Time_Series_Data or Time_Series_Data_Collection
        input data
    expandCategory : bool
        whether to expand category
    expandTime : bool
        whether to expand time
    preprocessType : ['ignore','pad','remove']
        preprocess data time across categories
    seperateLabels : bool
        whether to seperate labels and data
    version : int, optional
        feather version, by default 1
    chunksize : int
        size of feather file
    """
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