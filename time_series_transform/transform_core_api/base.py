import copy
import numpy as np
import pandas as pd
import pprint

class Time_Series_Data(object):

    def __init__(self):
        self.time_length = 0
        self._data = {}
        self._time_index= {}
        self._labels = {}

    @property
    def data(self):
        return self._data

    def set_data(self,inputData,label):
        if len(inputData) != self.time_length:
            raise ValueError('input data has different time length')
        self._data[label] = inputData

    @property
    def labels(self):
        return self._labels

    def set_labels(self,inputData,label):
        if len(inputData) != self.time_length:
            raise ValueError('input data has different time length')
        self._labels[label] = inputData

    @property
    def time_index(self):
        return self._time_index

    def set_time_index(self,inputData,label):
        self._time_index = {}
        self._time_index[label] = inputData
        self.time_length = len(inputData)

    def _get_dictionary_list_info(self,dictionary,indexSlice,label):
        res = {}
        if label is None:
            for i in dictionary:
                res[i] = dictionary[i][indexSlice]
        else:
            res[label] = dictionary[label][indexSlice]
        return res

    def __getitem__(self,ix):
        tmpInfo = self.labels
        tmpInfo.update(self.data)
        info = {}
        if isinstance(ix,tuple):
            t = ix[0]
            info.update(self._get_dictionary_list_info(self.time_index,t,None))
            for q in ix[1]:
                info.update(self._get_dictionary_list_info(tmpInfo,t,q))
        else:
            info.update(self._get_dictionary_list_info(self.time_index,ix,None))
            info.update(self._get_dictionary_list_info(tmpInfo,ix,None))
        return info

    def _reorder_list(self,sortingList,targetList,ascending):
        descending = 1-ascending
        ixList = sorted(range(len(sortingList)), key=lambda k: sortingList[k],reverse = descending)
        ordered_list = [targetList[i] for i in ixList]
        return ordered_list

    def sort(self,ascending):
        sortingList = list(self.time_index.values())[0]
        for data in self.data:
            self.data[data] = self._reorder_list(sortingList,self.data[data],ascending)
        for label in self.labels:
            self.labels[label] = self._reorder_list(sortingList,self.labels[label],ascending)
        for time in self.time_index:
            self.time_index[time] = self._reorder_list(sortingList,self.time_index[time],ascending)
        return self

    def make_dataframe(self):
        dfDict = {}
        dfDict.update(self.time_index)
        dfDict.update(self.labels)
        dfDict.update(self.data)
        return pd.DataFrame(dfDict)

    def transform(self,colName,newName,func,*args,**kwargs):
        pass


    def __repr__(self):
        dfDict = {}
        dfDict.update(self.time_index)
        dfDict.update(self.labels)
        dfDict.update(self.data)
        return str(dfDict)


class Time_Series_Data_Collection(object):
    def __init__(self):
        super().__init__()





class Time_Series_Tensor(object):
    def __init__(self,data,dtype,name):
        """
        Time_Series_Tensor the base class of this module
        
        this class define the basic object for managing data flow
        
        Parameters
        ----------
        data : numpy
            the data object has to be 1D numpy array
        dtype : numpy dtype
            this class defined the data type of the data
        name : str
            the name of the time series tensor
        """
        self.data = data
        self.dtype = dtype
        self.name = name

    def get_data_shape(self):
        """
        get_data_shape return the shape of data
        
        Returns
        -------
        tuple
            it will return the shape of the data
        """
        return self.data.shape
    
    def stack_time_series_tensors(self,time_series_tensor):
        """
        stack_time_series_tensors this function helps to stack other Time_Series_Tensor together
        
        this function helps to stack different sequences data together
        For example, stacking moving average with original sequence data.
        
        Parameters
        ----------
        time_series_tensor : Time_Series_Tensor
            the input Time_Series_Tensor's data will be stacked into the other
        """
        concatDim = time_series_tensor.data.ndim
        if time_series_tensor.data.ndim == 1:
            time_series_tensor.data = time_series_tensor.data.reshape((-1,1))

        if self.data.ndim == 1:
            self.data = self.data.reshape((-1,1))
        elif self.data.ndim == 2:
            self.data = self.data.reshape((self.data.shape[0],1,self.data.shape[1]))
        
        self.data = np.dstack((self.data,time_series_tensor.data))

        if concatDim == 1:
            self.data = self.data.reshape((-1,self.data.shape[-1]))

class Time_Series_Tensor_Factory(object):
    def __init__(self,data,tensorType):
        """
        Time_Series_Tensor_Factory this is the generative class for Time_Series_Tensor
        
        there are three different patterns of Time_Series_Tensor can be created
        1. sequence
            using window function, transform array into 3D tensor with 1 feature
        2. label
            skip window size of data to make label data for training set
        3. category
            clone single value and reshape array into (batch size, array length)

        Parameters
        ----------
        data : numpy array
            the data object has to be 1D numpy array
        tensorType : {'sequence','label','category'}
            the transformation type
        """
        super().__init__()
        self.tensorType = tensorType
        self.data = data

    def get_time_series_tensor(self,name,windowSize,seqSize,outType):
        """
        get_time_series_tensor the function to create Time_Series_Tensor
        
        Parameters
        ----------
        name : str
            the name of Time_Series_Tensor
        windowSize : int,
            the window size used for grouping sequence or label type of data, by default None
        seqSize : int,
            the batch size for category type of data, by default None
        outType : numpy data type,
            the data type of Time_Series_Tensor, by default None
        
        Returns
        -------
        Time_Series_Tensor
            it will turn a Time_Series_Tensor
        
        Raises
        ------
        ValueError
            if no valid tensoType input, this error will raise
        """
        if self.tensorType == 'sequence':
            # using window function, transform array into 3D tensor with 1 feature
            tensor = rolling_window(self.data,windowSize)
            tensor = tensor[:-1].reshape(-1,windowSize,1)
            return Time_Series_Tensor(tensor,outType,name)
        elif self.tensorType == 'label':
            # skip window size of data to make label data for training set
            tensor = self.data[windowSize:]
            return Time_Series_Tensor(tensor,outType,name)
        elif self.tensorType == 'category':
            # clone single value and reshape array into (batch size, array length)
            batchSize = seqSize - windowSize
            tensor = identity_window(self.data,batchSize)
            return Time_Series_Tensor(tensor,outType,name)
        elif self.tensorType == 'same':
            return Time_Series_Tensor(self.data,outType,name)
        else:
            raise ValueError('no value for tensorType')

class Time_Series_Dataset(object):
    def __init__(self,time_series_tensors):
        """
        Time_Series_Dataset comebine multiple Time_Series_Tensor and prepare the data for final output
        
        this class is prepared for generating training or testing data.
        
        Parameters
        ----------
        time_series_tensors : list of Time_Series_Tensor
            the list of Time_Series_Tensor for merging
        """
        super().__init__()
        self._time_series_tensors = time_series_tensors


    def make_dataset(self):
        """
        make_dataset combine the list of Time_Series_Tensor
        
        Returns
        -------
        dict
            dictionary of data and their dtypes
        """
        dataset = {}
        dtypes = {}
        for i in self._time_series_tensors:
            dataset[i.name] = i.data
            dtypes[i.name] = i.dtype
        return {'data':dataset,'dtypes':dtypes}
    

def rolling_window(arr, window):
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
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

def identity_window(arr,batchLen):
    """
    identity_window create an 2d numpy array with same items
    
    Parameters
    ----------
    arr : numpy array
        the original sequence
    batchLen : int
        window size
    
    Returns
    -------
    numpy 2d array
        2d array with same item corresponding to original sequence
    """
    res = None
    for value in arr:
        tmp = np.full((batchLen),value)
        tmp = tmp.reshape(-1,1)
        if res is None:
            res = tmp
        else:
            res = np.hstack([res,tmp])
    return res