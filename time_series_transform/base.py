import numpy as np

class Time_Series_Tensor(object):
    def __init__(self,data,dtype,name):
        self.data = data
        self.dtype = dtype
        self.name = name

    def get_data_shape(self):
        return self.data.shape
    
    def stack_time_series_tensors(self,time_series_tensor):
        self.data = np.dstack((self.data,time_series_tensor.data))

class Time_Series_Tensor_Factory(object):
    
    def __init__(self,data,tensorType):
        super().__init__()
        self.tensorType = tensorType
        self.data = data

    def get_time_series_tensor(self,name,windowSize,batchSize=None,outType=None):
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
            tensor = identity_window(self.data,batchSize)
            return Time_Series_Tensor(tensor,outType,name)
        else:
            raise ValueError('no value for tensorType')

class Time_Series_Dataset(object):
    def __init__(self,time_series_tensors):
        super().__init__()
        self._time_series_tensors = time_series_tensors


    def make_dataset(self):
        dataset = {}
        for i in self._time_series_tensors:
            dataset[i.name] = i.data
        return dataset
    

def rolling_window(arr, window):
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

def identity_window(arr,batchLen):
    res = None
    for value in arr:
        tmp = np.full((batchLen),value)
        tmp = tmp.reshape(-1,1)
        if res is None:
            res = tmp
        else:
            res = np.hstack([res,tmp])
    return res