import numpy as np
import pandas as pd
from time_series_transform.io.base import io_base
from time_series_transform.transform_core_api.base import (
    Time_Series_Data_Collection,
    Time_Series_Data
    )


class Numpy_IO (io_base):
    def __init__(self, time_series, timeSeriesColIx, mainCategoryColIx):
        """
        Numpy_IO IO class for numpy data
        
        Parameters
        ----------
        time_series : Time_Series_Data or Time_Series_Data_Collection
            input data
        timeSeriesCol : str or int
            index of time period column
        mainCategoryCol : str of int
            index of category column
        """
        super().__init__(time_series, timeSeriesColIx, mainCategoryColIx)
        if self.dictList is not None:
            self.dictList = {}
            for i in range(len(time_series.T)):
                self.dictList[i] = time_series.T[i]

    def from_numpy (self):
        """
        from_numpy 
        transform numpy ndArray to Time_Series_Data or Time_Series_Data_Collection
        Returns
        -------
        Time_Series_Data or Time_Series_Data_Collection
        """
        if self.mainCategoryCol is None:
            return self.to_single()
        return self.to_collection()


    def to_numpy(self,expandTime,expandCategory,preprocessType,labelList):
        """
        to_numpy transform Time_Series_Data or Time_Series_Data_Collection
        to numpy ndArray
        
        Parameters
        ----------
        expandCategory : bool
            whether to expand category
        expandTime : bool
            whether to expand time
        preprocessType : ['ignore','pad','remove']
            preprocess data time across categories
        labelList : list
            label list
        
        Returns
        -------
        numpy ndArray
        """
        if isinstance(self.time_series,Time_Series_Data):
            data = self.from_single(expandTime)
        if isinstance(self.time_series,Time_Series_Data_Collection):
            data = self.from_collection(expandCategory,expandTime,preprocessType)
        for i in data:
            if isinstance(data[i],np.ndarray):
                data[i] = data[i].tolist()
        if labelList is None:
            return pd.DataFrame(data).values
        labelDict = {}
        dataDict = {}
        print(f"label {labelList}")
        for i in data:
            if i in labelList:
                labelDict[i] = data[i]
                continue
            dataDict[i] = data[i]
        return pd.DataFrame(dataDict).values,pd.DataFrame(labelDict).values


def from_numpy(numpyArray,timeSeriesCol,mainCategoryCol=None):
    """
    from_numpy transform numpy ndArray
         to Time_Series_Data or Time_Series_Data_Collection
    
    Parameters
    ----------
    numpyArray : numpy ndArray
        input data
    timeSeriesCol : str or int
        index of time period column
    mainCategoryCol : str of int
        index of category column
    
    Returns
    -------
    Time_Series_Data or Time_Series_Data_Collection
    
    Raises
    ------
    ValueError
        invalid input data
    """
    if not isinstance(numpyArray,np.ndarray):
        raise ValueError('input data must be numpy array')
    numpyio = Numpy_IO(numpyArray,timeSeriesCol,mainCategoryCol)
    return numpyio.from_numpy()

def to_numpy(time_series_data,expandCategory,expandTime,preprocessType,seperateLabels=False):
    """
    to_numpy 
    
    transform Time_Series_Data or Time_Series_Data_Collection
    to numpy ndArray
    
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
    [type]
        [description]
    
    Raises
    ------
    ValueError
        [description]
    """
    labelsList = []
    if isinstance(time_series_data,Time_Series_Data):
        numpyio = Numpy_IO(time_series_data,time_series_data.time_seriesIx,None)
        expandCategory = None
        labelsList = list(time_series_data.labels.keys())
    elif isinstance(time_series_data,Time_Series_Data_Collection):
        numpyio = Numpy_IO(
            time_series_data,
            time_series_data._time_series_Ix,
            time_series_data._categoryIx
            )
        for i in time_series_data:
            print(time_series_data[i].labels)
            labelsList.extend(list(time_series_data[i].labels.keys()))
            labelsList = list(set(labelsList))
    else:
        raise ValueError('input data should be time_series_data or time_series_collection')
    if seperateLabels == False:
        return numpyio.to_numpy(expandTime,expandCategory,preprocessType,None)
    return numpyio.to_numpy(expandTime,expandCategory,preprocessType,labelsList)



__all__= [
    'from_numpy',
    'to_numpy'
]