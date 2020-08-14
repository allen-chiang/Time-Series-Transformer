import numpy as np

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