import gc
import uuid
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq
from collections import defaultdict
from time_series_transform import io
from time_series_transform.transform_core_api.base import (Time_Series_Data,Time_Series_Data_Collection)
from time_series_transform.plot import *

class Time_Series_Transformer(object):

    def __init__(self,data,timeSeriesCol,mainCategoryCol=None):
        """
        __init__ the class for time series data manipulation
        
        it can perform different data manipulation: making lag data,
        lead data, lag sequence data, or do a customize data manipulation.
        It also built in native plot and io functions. IO function currently
        support pandas DataFrame, numpy ndArray, apache arrow table , apache feather,
        and apache parquet
        
        Parameters
        ----------
        data : dict of list, Time_Series_Data, or Time_Series_Collection
            the value of data.
        timeSeriesCol : str
            the time series period column of the data. For example, time or date
        mainCategoryCol : str or None
            the main category column of the time series data
            for example, symbol ticker for stock data. Or, the product segment for inventory
        """
        super().__init__()
        if isinstance(data,(Time_Series_Data,Time_Series_Data_Collection)):
            self.time_series_data = data
        else:
            self.time_series_data = self._setup_time_series_data(data,timeSeriesCol,mainCategoryCol)
        self.timeSeriesCol = timeSeriesCol
        self._isCollection = [True if mainCategoryCol is not None else False][0]
        self.mainCategoryCol = mainCategoryCol
        self.plot = TimeSeriesPlot(self.time_series_data)

    def _setup_time_series_data(self,data,timeSeriesCol,mainCategoryCol):
        if timeSeriesCol is None:
            raise KeyError("time series index is required")
        tsd = Time_Series_Data(data,timeSeriesCol)
        if mainCategoryCol is None:
            return tsd
        tsc = Time_Series_Data_Collection(tsd,timeSeriesCol,mainCategoryCol)
        return tsc
    
    def transform(self,inputLabels,newName,func,n_jobs =1,verbose = 0,backend='loky',*args,**kwargs):
        """
        transform the wrapper of functions performing data manipulation
        
        This function provides a way to do different data manipulation.
        The output data should be either pandas dataFrame, numpy ndArray, or list of dict.
        Also, the data should have the same time length as the original data.
        
        Parameters
        ----------
        inputLabels : str, numeric data or list of data or numeric data
            the input data columns passing to function
        newName : str
            the output data name or prefix
            if the out function provides the new name, it will automatically become prefix
        func : function
            the data manipulation function
        n_jobs : int, optional
            joblib implemention, only used when mainCategoryCol is given, by default 1
        verbose : int, optional
            joblib implmentation only used when mainCategoryCol is given, by default 0
        backend : str, optional
            joblib implmentation only used when mainCategoryCol is given, by default 'loky'
        
        Returns
        -------
        self
        """
        if isinstance(self.time_series_data,Time_Series_Data_Collection):
            self.time_series_data = self.time_series_data.transform(inputLabels,newName,func,n_jobs =1,verbose = 0,backend='loky',*args,**kwargs)
        else:
            self.time_series_data = self.time_series_data.transform(inputLabels,newName,func,*args,**kwargs)
        return self


    def _transform_wrapper(self,inputLabels,newName,func,suffix,suffixNum,inputAsList,n_jobs,verbose,*args,**kwargs):
        if isinstance(inputLabels,list) == False:
            inputLabels = [inputLabels]
        if self._isCollection:
            if inputAsList == False:
                for i in inputLabels:
                    labelName = [f'{i}{suffix}{str(suffixNum)}' if suffix is not None else f"{i}{str(suffixNum)}"][0]
                    self.time_series_data.transform(i,labelName,func,n_jobs =n_jobs,verbose = verbose,*args,**kwargs)
                return
            labelName = newName
            self.time_series_data.transform(inputLabels,labelName,func,n_jobs =n_jobs,verbose = verbose,*args,**kwargs)
        else:
            if inputAsList == False:
                for i in inputLabels:
                    labelName = [f'{i}{suffix}{str(suffixNum)}' if suffix is not None else f"{i}{str(suffixNum)}"][0]
                    self.time_series_data.transform(i,labelName,func,*args,**kwargs)
                return
            labelName = newName
            self.time_series_data.transform(inputLabels,labelName,func,*args,**kwargs)


    def make_lag(self,inputLabels,lagNum,suffix=None,fillMissing=np.nan,verbose=0,n_jobs=1):
        """
        make_lag making lag data for a given list of data
        
        Parameters
        ----------
        inputLabels : str, numeric or list of str, or numeric
            the name of input data 
        lagNum : int
            the target lag period to make
        suffix : str, optional
            the suffix of new data, by default None
        fillMissing : object, optional
            the data for filling missing data, by default np.nan
        verbose : int, optional
            joblib implmentation only used when mainCategoryCol is given, by default 0
        n_jobs : int, optional
            joblib implmentation only used when mainCategoryCol is given, by default 1
        
        Returns
        -------
        self
        """
        self._transform_wrapper(
            inputLabels,
            None,
            make_lag,
            suffix,
            lagNum,
            False,
            n_jobs,
            verbose,
            lagNum=lagNum,
            fillMissing=fillMissing
            )
        return self

    def make_lead(self,inputLabels,leadNum,suffix=None,fillMissing=np.nan,verbose=0,n_jobs=1):
        """
        make_lead make_lead making lead data for a given list of data
        
        Parameters
        ----------
        inputLabels : str, numeric or list of str, or numeric
            the name of input data 
        leadNum : int
            the target lead period to make
        suffix : str, optional
            the suffix of new data, by default None
        fillMissing : object, optional
            the data for filling missing data, by default np.nan
        verbose : int, optional
            joblib implmentation only used when mainCategoryCol is given, by default 0
        n_jobs : int, optional
            joblib implmentation only used when mainCategoryCol is given, by default 1
        
        Returns
        -------
        self
        """
        self._transform_wrapper(
            inputLabels,
            None,
            make_lead,
            suffix,
            leadNum,
            False,
            n_jobs,
            verbose,
            leadNum=leadNum,
            fillMissing=fillMissing
            )
        return self
                
    def make_lag_sequence(self,inputLabels,windowSize,lagNum,suffix=None,fillMissing=np.nan,verbose=0,n_jobs=1):
        """
        make_lag_sequence making lag sequence data 
        
        this function could be useful for deep learning.
        
        Parameters
        ----------
        inputLabels : str, numeric or list of str, or numeric
            the name of input data 
        windowSize : int
            the length of sequence
        lagNum : int
            the lag period of sequence
        suffix : str, optional
            the suffix of new data, by default None
        fillMissing : object, optional
            the data for filling missing data, by default np.nan
        verbose : int, optional
            joblib implmentation only used when mainCategoryCol is given, by default 0
        n_jobs : int, optional
            joblib implmentation only used when mainCategoryCol is given, by default 1
        
        Returns
        -------
        self
        """
        self._transform_wrapper(
            inputLabels,
            None,
            make_lag_sequnece,
            suffix,
            windowSize,
            False,
            n_jobs,
            verbose,
            windowSize=windowSize,
            lagNum = lagNum,
            fillMissing=fillMissing
            )
        return self

    def make_lead_sequence(self,inputLabels,windowSize,leadNum,suffix=None,fillMissing=np.nan,verbose=0,n_jobs=1):
        """
        make_lead_sequence making lead sequence data 
        
        this function could be useful for deep learning.
        
        Parameters
        ----------
        inputLabels : str, numeric or list of str, or numeric
            the name of input data 
        windowSize : int
            the length of sequence
        leadNum : int
            the lead period of sequence
        suffix : str, optional
            the suffix of new data, by default None
        fillMissing : object, optional
            the data for filling missing data, by default np.nan
        verbose : int, optional
            joblib implmentation only used when mainCategoryCol is given, by default 0
        n_jobs : int, optional
            joblib implmentation only used when mainCategoryCol is given, by default 1
        
        Returns
        -------
        self
        """
        self._transform_wrapper(
            inputLabels,
            None,
            lead_sequence,
            suffix,
            windowSize,
            False,
            n_jobs,
            verbose,
            windowSize=windowSize,
            leadNum=leadNum,
            fillMissing=fillMissing
            )
        return self

    def make_identical_sequence(self,inputLabels,windowSize,suffix=None,verbose=0,n_jobs=1):
        """
        make_identical_sequence making sequences having same data
        
        this function will make same data for a givne sequence.
        it could be useful for category data in deep learning.
        
        Parameters
        ----------
        inputLabels : str, numeric or list of str, or numeric
            the name of input data 
        windowSize : int
            the length of sequence
        suffix : str, optional
            the suffix of new data, by default None
        verbose : int, optional
            joblib implmentation only used when mainCategoryCol is given, by default 0
        n_jobs : int, optional
            joblib implmentation only used when mainCategoryCol is given, by default 1
        
        Returns
        -------
        self
        """
        self._transform_wrapper(
            inputLabels,
            None,
            identity_window,
            suffix,
            windowSize,
            False,
            n_jobs,
            verbose,
            windowSize=windowSize
            )
        return self

    def make_stack_sequence(self,inputLabels,newName,axis =-1,verbose=0,n_jobs=1):
        """
        make_stack_sequence stacking sequences data
        
        making multiple seqeunce data into one on the given axis
        
        Parameters
        ----------
        inputLabels : str, numeric or list of str, or numeric
            the name of input data 
        newName : str
            new name for the stacking data
        axis : int, optional
            the axis for stacking (numpy stack implmentation), by default -1
        verbose : int, optional
            joblib implmentation only used when mainCategoryCol is given, by default 0
        n_jobs : int, optional
            joblib implmentation only used when mainCategoryCol is given, by default 1
        
        Returns
        -------
        [type]
            [description]
        """
        self._transform_wrapper(
            inputLabels,
            newName,
            stack_sequence,
            None,
            '',
            True,
            n_jobs,
            verbose,
            axis =axis
            )
        return self


    def make_label(self,key,collectionKey=None):
        """
        make_label make label data
        
        it will turn the data into label.
        when using io functions, specifing sepLabel parameter can seperate label and data.
        
        Parameters
        ----------
        key : str or numeric data
            the target data name
        collectionKey : str or numeric data, optional
            the target collection, if None, all collection is selected, by default None
        
        Returns
        -------
        self
        """
        if self._isCollection:
            if collectionKey is None:
                for i in self.time_series_data:
                    data = self.time_series_data[i][:,[key]][key]
                    self.time_series_data[i].set_labels(data,key)
                    self.time_series_data[i].remove(key,'data')
            else:
                data = self.time_series_data[collectionKey][:,[key]][key]
                self.time_series_data[collectionKey].set_labels(data,key)
                self.time_series_data[collectionKey].remove(key,'data')
        else:
            data = self.time_series_data[:,[key]][key]
            self.time_series_data.set_labels(data,key)
            self.time_series_data.remove(key,'data')
        return self

    def remove_different_category_time(self):
        """
        remove_different_category_time 
        remove different time index for category
        if mainCategoryCol is not specified, this function has no function.
        Returns
        -------
        self
        """
        if self._isCollection:
            self.time_series_data.remove_different_time_index()
        else:
            warnings.warn('Setup mainCategoryCol is necessary for this function')
        return self

    def pad_different_category_time(self,fillMissing= np.nan):
        """
        pad_different_category_time 
        pad time length
        if mainCategoryCol is not specified, this function has no function.
        
        Parameters
        ----------
        fillMissing : object, optional
            data for filling paded data, by default np.nan
        
        Returns
        -------
        self
        """
        if self._isCollection:
            self.time_series_data.pad_time_index(fillMissing)
        else:
            warnings.warn('Setup mainCategoryCol is necessary for this function')
        return self

    def remove_category(self,categoryName):
        """
        remove_category remove a specific category data
        
        Parameters
        ----------
        categoryName : str or numeric data
            the target category to be removed
        
        Returns
        -------
        self
        """
        if self._isCollection:
            self.time_series_data.remove(categoryName)
        return self

    def remove_feature(self,colName):
        """
        remove_feature remove certain data or labels
        
        Parameters
        ----------
        colName : str or numeric
            target column or data to be removed
        
        Returns
        -------
        self
        """
        if isinstance(self.time_series_data,Time_Series_Data_Collection):
            for i in self.time_series_data:
                self.time_series_data[i].remove(colName)
                return self
        self.time_series_data.remove(colName)
        return self

    def dropna(self,categoryKey=None):
        """
        dropna drop null values
        
        remove null values for all or a specific category
        
        Parameters
        ----------
        categoryKey :  str or numeric, optional
            if None all category will be chosen, by default None
        
        Returns
        -------
        self
        """
        if isinstance(self.time_series_data,Time_Series_Data):
            self.time_series_data = self.time_series_data.dropna()
            return self
        self.time_series_data = self.time_series_data.dropna(categoryKey)
        return self


    @classmethod
    def from_pandas(cls, pandasFrame,timeSeriesCol,mainCategoryCol):
        """
        from_pandas import data from pandas dataFrame
        
        Parameters
        ----------
        pandasFrame : pandas DataFrame
            input data
        timeSeriesCol : str or numeric
            time series column name
        mainCategoryCol : str or numeric
            main category name
        
        Returns
        -------
        Time_Series_Transformer
        """
        data = io.from_pandas(pandasFrame,timeSeriesCol,mainCategoryCol)
        return cls(data,timeSeriesCol,mainCategoryCol)

    @classmethod
    def from_numpy(cls,numpyData,timeSeriesCol,mainCategoryCol):
        """
        from_numpy import data from numpy
        
        Parameters
        ----------
        numpyData : numpy ndArray
            input data
        timeSeriesCol : int
            index of time series column
        mainCategoryCol : int
            index of main category column
        
        Returns
        -------
        Time_Series_Transformer
        """
        data = io.from_numpy(numpyData,timeSeriesCol,mainCategoryCol)
        return cls(data,timeSeriesCol,mainCategoryCol)

    @classmethod
    def from_feather(cls,feather_dir,timeSeriesCol,mainCategoryCol,columns=None):
        """
        from_feather import data from feather

        Parameters
        ----------
        feather_dir : str
            directory of feather file
        timeSeriesCol : str or numeric
            time series column name
        mainCategoryCol : str or numeric
            main category name
        columns : str or numeric, optional
            target columns (apache arrow implmentation), by default None
        
        Returns
        -------
        Time_Series_Transformer
        """
        data = io.from_feather(
            feather_dir,
            timeSeriesCol,
            mainCategoryCol,
            columns
            )
        return cls(data,timeSeriesCol,mainCategoryCol)
    
    @classmethod
    def from_parquet(cls,parquet_dir,timeSeriesCol,mainCategoryCol,columns = None,partitioning='hive',filters=None,filesystem=None):
        """
        from_parquet import data from parquet file
        
        Parameters
        ----------
        parquet_dir : str
            directory of parquet file
        timeSeriesCol : str or numeric
            time series column name
        mainCategoryCol : str or numeric
            main category name
        columns : str or numeric, optional
            target columns (apache arrow implmentation), by default None
        partitioning : str, optional
            type of partitioning, by default 'hive'
        filters : str, optional
            filter (apache arrow implmentation), by default None
        filesystem : str, optional
            filesystem (apache arrow implmentation), by default None
        
        Returns
        -------
        Time_Series_Transformer
        """
        data = io.from_parquet(
            parquet_dir,
            timeSeriesCol,
            mainCategoryCol,
            columns,
            partitioning,
            filters,
            filesystem
            )
        return cls(data,timeSeriesCol,mainCategoryCol)
    
    @classmethod
    def from_arrow_table(cls,arrow_table,timeSeriesCol,mainCategoryCol):
        """
        from_arrow_table import data from apache arrow table
        
        Parameters
        ----------
        arrow_table : arrow table
            input data
        timeSeriesCol : str or numeric
            time series column name
        mainCategoryCol : str or numeric
            main category name
        
        Returns
        -------
        Time_Series_Transformer
        """
        data = io.from_arrow_table(arrow_table,timeSeriesCol,mainCategoryCol)
        return cls(data,timeSeriesCol,mainCategoryCol)

    def to_feather(self,dirPaths,expandCategory=False,expandTime=False,preprocessType='ignore',sepLabel = False,version = 1,chunksize=None):
        """
        to_feather output data into feather format
        
        Parameters
        ----------
        dirPaths : str
            directory of output data
        expandCategory : bool, optional
            whether to expand category, by default False
        expandTime : bool, optional
            whether to expand time index column, by default False
        preprocessType : {'ignore','pad','remove'}, optional
            the preprocessing type before out data, by default 'ignore'
        sepLabel : bool, optional
            whether to seperate label data, by default False
        version : int, optional
            fether version (apache arrow implmentation), by default 1
        chunksize : int, optional
            chunksize for output (apache arrow implmentation), by default None
        
        """
        return io.to_feather(
            dirPaths= dirPaths,
            time_series_data= self.time_series_data,
            expandCategory = expandCategory,
            expandTime = expandTime,
            preprocessType = preprocessType,
            seperateLabels= sepLabel,
            version= version,
            chunksize= chunksize
        )
    
    def to_parquet(self,dirPaths,expandCategory=False,expandTime=False,preprocessType='ignore',sepLabel = False,version = '1.0',isDataset=False,partition_cols= None):
        """
        to_parquet output data into parquet format
        
        Parameters
        ----------
        dirPaths : str
            directory of output data
        expandCategory : bool, optional
            whether to expand category, by default False
        expandTime : bool, optional
            whether to expand time index column, by default False
        preprocessType : {'ignore','pad','remove'}, optional
            the preprocessing type before out data, by default 'ignore'
        sepLabel : bool, optional
            whether to seperate label data, by default False
        version : str, optional
            parquet version (apache arrow implmentation), by default '1.0'
        isDataset : bool, optional
            whether to output data as dataset format (apache arrow implmentation), by default False
        partition_cols : str, optional
            whether to partition data (apache arrow implmentation), by default None
        
        """
        return io.to_parquet(
            dirPaths= dirPaths,
            time_series_data= self.time_series_data,
            expandCategory=expandCategory,
            expandTime =expandTime,
            preprocessType= preprocessType,
            seperateLabels= sepLabel,
            version = version,
            isDataset = isDataset,
            partition_cols = partition_cols
        )

    def to_arrow_table(self,expandCategory=False,expandTime=False,preprocessType='ignore',sepLabel = False):
        """
        to_arrow_table output data as apache arrow table format
        
        Parameters
        ----------
        expandCategory : bool, optional
            whether to expand category, by default False
        expandTime : bool, optional
            whether to expand time index column, by default False
        preprocessType : {'ignore','pad','remove'}, optional
            the preprocessing type before out data, by default 'ignore'
        sepLabel : bool, optional
            whether to seperate label data, by default False
        
        Returns
        -------
        arrow table
        """
        return io.to_arrow_table(
            time_series= self.time_series_data,
            expandCategory= expandCategory,
            expandTime= expandTime,
            preprocessType = preprocessType,
            seperateLabels= sepLabel
        )

    def to_pandas(self,expandCategory=False,expandTime=False,preprocessType='ignore',sepLabel = False):
        """
        to_pandas output data into pandas dataFrame
        
        Parameters
        ----------
        expandCategory : bool, optional
            whether to expand category, by default False
        expandTime : bool, optional
            whether to expand time index column, by default False
        preprocessType : {'ignore','pad','remove'}, optional
            the preprocessing type before out data, by default 'ignore'
        sepLabel : bool, optional
            whether to seperate label data, by default False

        Returns
        -------
        pandas dataFrame

        """
        return io.to_pandas(
            self.time_series_data,
            expandCategory = expandCategory,
            expandTime = expandTime,
            preprocessType=preprocessType,
            seperateLabels = sepLabel
            )
        
    def to_numpy(self,expandCategory=False,expandTime=False,preprocessType='ignore',sepLabel = False):
        """
        to_numpy output data into numpy format
        
        Parameters
        ----------
        expandCategory : bool, optional
            whether to expand category, by default False
        expandTime : bool, optional
            whether to expand time index column, by default False
        preprocessType : {'ignore','pad','remove'}, optional
            the preprocessing type before out data, by default 'ignore'
        sepLabel : bool, optional
            whether to seperate label data, by default False
        
        Returns
        -------
        numpy ndArray
        """
        return io.to_numpy(self.time_series_data,expandCategory,expandTime,preprocessType,sepLabel)

    def to_dict(self):
        """
        to_dict output data as dictionary list
        
        Returns
        -------
        dict of list
        """
        return self.time_series_data[:]

    def __eq__(self,other):
        if isinstance(other,Time_Series_Transformer):
            return self.time_series_data == other.time_series_data
        return False


    def _statement_maker(self,tsd,mainCategory):
        dataCol = list(tsd._get_all_info().keys())
        timeLength = tsd.time_length
        statement = "data column\n-----------\n"
        for i in dataCol:
            statement += f"{i}\n"
        statement += f"time length: {str(timeLength)}\n"
        statement += f"category: {str(mainCategory)}\n\n"
        return statement


    def __repr__(self):
        if isinstance(self.time_series_data,Time_Series_Data):
            return self._statement_maker(self.time_series_data,self.mainCategoryCol)
        statement = ""
        for i in self.time_series_data:
            statement+= self._statement_maker(self.time_series_data[i],i)
        statement += f"main category column: {self.mainCategoryCol}"
        return statement

def make_sequence(arr, window,fillMissing=np.nan):
    """
    rolling_window create an rolling window tensor
    
    this function create a rolling window numpy tensor given its original sequence and window size
    
    Parameters
    ----------
    arr : numpy 1D array
        the original data sequence
    window : int
        aggregation window size
    
    Returns
    -------
    numpy 2d array
        the rolling window array
    """
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    seq = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    empty = np.empty(((len(arr)-seq.shape[0],seq.shape[1])))
    empty[:] = fillMissing
    res = np.vstack([empty,seq])
    return res


def make_lag_sequnece(data,windowSize,lagNum,fillMissing):
    lagdata = np.array(make_lag(data,lagNum,fillMissing))
    return make_sequence(lagdata,windowSize,fillMissing)

def identity_window(arr,windowSize):
    return np.repeat(arr,windowSize).reshape((-1,windowSize))

def make_lead(data,leadNum,fillMissing):
    res = np.empty((leadNum))
    res[:] = fillMissing
    res = res.tolist()        
    leadValues = data[leadNum:].tolist()
    leadValues.extend(res)        
    return leadValues        

def make_lag(data,lagNum,fillMissing):
    res = np.empty((lagNum))
    res[:] = fillMissing
    res = res.tolist()        
    lagValues = data[:-lagNum]
    res.extend(lagValues)        
    return res

def lead_sequence(arr,leadNum,windowSize,fillMissing=np.nan):
    shape = arr.shape[:-1] + (arr.shape[-1] - windowSize + 1, windowSize)
    strides = arr.strides + (arr.strides[-1],)
    seq = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    seq = seq[leadNum:]
    empty = np.empty(((len(arr)-seq.shape[0],seq.shape[1])))
    empty[:] = fillMissing
    res = np.vstack([seq,empty])
    return res

def stack_sequence(arrDict, axis = -1):
    res = []
    for ix, v in enumerate(arrDict):
        data = np.array(arrDict[v])
        res.append(data)
    res = np.stack(res,axis = axis )
    return res


__all__=[
    "Time_Series_Transformer"
]