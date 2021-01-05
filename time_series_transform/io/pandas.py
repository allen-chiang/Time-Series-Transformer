import pandas as pd
from time_series_transform.transform_core_api.base import Time_Series_Data, Time_Series_Data_Collection
from time_series_transform.io.base import io_base
import numpy as np

class Pandas_IO (io_base):
    def __init__(self, time_series, timeSeriesCol, mainCategoryCol):
        """
        Pandas_IO IO class for pandas dataFrame
        
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
    
    def from_pandas(self):
        """
        from_pandas transform dataFrame to 
        Time_Series_Data or Time_Series_Data_Collection
        
        Returns
        -------
        Time_Series_Data or Time_Series_Data_Collection
        
        Raises
        ------
        ValueError
            invalid data input
        """
        if not isinstance(self.dictList,pd.DataFrame):
            raise ValueError("input data must be pandas frame")
        if self.mainCategoryCol is None:
            return self.to_single()
        return self.to_collection()
        

    def to_pandas(self,expandTime,expandCategory,preprocessType):
        """
        to_pandas transform Time_Series_Data or Time_Series_Data_Collection
        into pandas dataFrame
        
        Parameters
        ----------
        expandCategory : bool
            whether to expand category
        expandTime : bool
            whether to expand time
        preprocessType : ['ignore','pad','remove']
            preprocess data time across categories
        
        Returns
        -------
        pandas dataFrame
        
        Raises
        ------
        ValueError
            invalid data type
        """
        if isinstance(self.time_series,Time_Series_Data):
            data = self.from_single(expandTime)
            for i in data:
                if isinstance(data[i],np.ndarray):
                    data[i] = data[i].tolist()
            return pd.DataFrame(data)
        if isinstance(self.time_series,Time_Series_Data_Collection):
            data = self.from_collection(expandCategory,expandTime,preprocessType)
            for i in data:
                if isinstance(data[i],np.ndarray):
                    data[i] = data[i].tolist()
            return pd.DataFrame(data)
        raise ValueError("Invalid data type")


def from_pandas(pandasFrame,timeSeriesCol,mainCategoryCol=None):
    """
    from_pandas         
    from_pandas transform dataFrame to 
    Time_Series_Data or Time_Series_Data_Collection
    
    Parameters
    ----------
    pandasFrame : pandas dataFrame
        input data
    timeSeriesCol : str or int
        index of time period column
    mainCategoryCol : str of int
        index of category column
    
    Returns
    -------
    Time_Series_Data or Time_Series_Data_Collection
    """
    pio = Pandas_IO(pandasFrame,timeSeriesCol,mainCategoryCol)
    return pio.from_pandas()

def to_pandas(time_series_data,expandCategory,expandTime,preprocessType,seperateLabels = False):
    """
    to_pandas 
    transform Time_Series_Data or Time_Series_Data_Collection
    into pandas dataFrame
    
    Parameters
    ----------
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
    
    Returns
    -------
    pandas dataFrame
    
    Raises
    ------
    ValueError
        invalid data input
    """
    labelsList = []
    if isinstance(time_series_data,Time_Series_Data):
        pio = Pandas_IO(time_series_data,time_series_data.time_seriesIx,None)
        expandCategory = None
        preprocessType = None
        labelsList = list(time_series_data.labels.keys())
    elif isinstance(time_series_data,Time_Series_Data_Collection):
        pio = Pandas_IO(
            time_series_data,
            time_series_data._time_series_Ix,
            time_series_data._categoryIx
            )
        labelsList = []
        for i in time_series_data:
            labelsList.extend(list(time_series_data[i].labels.keys()))
            labelsList = list(set(labelsList))
    else:
        raise ValueError('Input data should time_series_data or time_series_collection')
    df = pio.to_pandas(expandTime,expandCategory,preprocessType)
    if seperateLabels == False:
        return df
    return df.drop(labelsList,axis =1),df[labelsList]
    

__all__ = [
    'from_pandas',
    'to_pandas'
]