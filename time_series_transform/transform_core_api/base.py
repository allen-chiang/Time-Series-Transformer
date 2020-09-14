import copy
import numpy as np
import pandas as pd
import pprint
from joblib import Parallel, delayed
from collections import ChainMap
from collections import Counter

class Time_Series_Data(object):

    def __init__(self):
        self.time_length = 0
        self._data = {}
        self._time_index= {}
        self._labels = {}

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def time_index(self):
        return self._time_index

    def set_data(self,inputData,label):
        if len(inputData) != self.time_length:
            raise ValueError('input data has different time length')
        self._data[label] = np.array(inputData)
        return self


    def set_labels(self,inputData,label):
        if len(inputData) != self.time_length:
            raise ValueError('input data has different time length')
        self._labels[label] = np.array(inputData)
        return self


    def set_time_index(self,inputData,label):
        self._time_index = {}
        self._time_index[label] = np.array(inputData)
        self.time_length = len(inputData)
        return self

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
        return np.array(ordered_list)

    def sort(self,ascending=True):
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

    def _single_transform(self,colName,func,*args,**kwargs):
        if colName in self.data:
            arr = self.data[colName]
            return func(arr,*args,**kwargs),'data'
        else:
            arr = self.labels[colName]
            return func(arr,*args,**kwargs),'labels'

    def _list_transform(self,inputList,func,*args,**kwargs):
        arrDict = {}
        outputType = 'label'
        for col in inputList:
            if col in self.data:
                arrDict[col] = self.data[col]
                outputType='data'
            else:
                arrDict[col] = self.labels[col]
        arrDict = func(arrDict,*args,**kwargs)
        return arrDict,outputType

    def transform(self,inputLabels,newName,func,*args,**kwargs):
        # transform
        if isinstance(inputLabels,list):
            arr,outputType = self._list_transform(inputLabels,func,*args,**kwargs)
        else:
            arr,outputType = self._single_transform(inputLabels,func,*args,**kwargs)

        # organize into dict
        if isinstance(arr,pd.DataFrame):
            arr = arr.to_dict(orient='list')
            arr = { f"{newName}_{k}": v for k, v in arr.items() }
        elif isinstance(arr,list) or isinstance(arr,np.ndarray):
            arr = {newName:np.array(arr)}   
        elif isinstance(arr,pd.Series):
            arr = {newName:arr.values}

        if outputType == 'data':
            self._data.update(arr)
        else:
            self._labels.update(arr)
        # update existing dict
        return self
        

    def _get_all_info(self):
        dfDict = {}
        dfDict.update(self.time_index)
        dfDict.update(self.labels)
        dfDict.update(self.data)        
        return dfDict

    def __repr__(self):
        return str(self._get_all_info())

    def __eq__(self, other):
        left = self._get_all_info()
        right = other._get_all_info()
        if len(left) != len(right):
            return False
        for i in left:
            if i not in right:
                return False
            left[i] = list(left[i])
            right[i] = list(right[i])
        return left == right

        
class Time_Series_Data_Colleciton(object):
    def __init__(self,time_series_data,time_seriesIx,categoryIx):
        super().__init__()
        self._time_series_data_collection = self._expand_time_series_data(time_series_data,categoryIx)
        self._time_series_Ix = time_seriesIx
        self._categoryIx = categoryIx

    @property
    def time_series_data_collection(self):
        return self._time_series_data_collection


    def _expand_time_series_data(self,time_series_data,categoryIx):
        dct = {}
        for i in list(set(time_series_data[:,[categoryIx]][categoryIx])):
            ixList = np.where(time_series_data[:,[categoryIx]][categoryIx]==i)
            tmp = {}
            tmp = Time_Series_Data()
            for t in time_series_data.time_index:
                tmp.set_time_index(time_series_data.time_index[t][ixList],t)
            for d in time_series_data.data:
                tmp.set_data(time_series_data.data[d][ixList],d)
            for l in time_series_data.labels:
                if l == categoryIx:
                    continue
                tmp.set_data(time_series_data.labels[l][ixList],l)
            dct[i] = tmp
        return dct


    def _parallel_transform(self,category,time_series_data,inputLabels,newName,func,*args,**kwargs):
        return {category:time_series_data.transform(inputLabels,newName,func,*args,**kwargs)}


    def transform(self,inputLabels,newName,func,n_jobs =1,verbose = 0,backend='loky',*args,**kwargs):
        dctList= Parallel(n_jobs=n_jobs,verbose = verbose, backend=backend)(delayed(self._parallel_transform)(
            c,self._time_series_data_collection[c],inputLabels,newName,func,*args,**kwargs
            ) for c in self.time_series_data_collection)
        self._time_series_data_collection = dict(ChainMap(*dctList))
        return self

    def remove_different_time_index(self):
        timeix = []
        for i in self._time_series_data_collection:
            timeix.extend(self._time_series_data_collection[i][:][self._time_series_Ix])
        timeix = Counter(timeix)
        timeCol = [k for k,v in timeix.items() if v == len(self._time_series_data_collection)]    
        for i in self._time_series_data_collection:
            tmp_time = Time_Series_Data()
            ix = np.isin(self._time_series_data_collection[i][:][self._time_series_Ix],timeCol)
            for t in self._time_series_data_collection[i].time_index:
                tmp = self._time_series_data_collection[i].time_index[t][ix]
                tmp_time.set_time_index(tmp,t)
            for d in self._time_series_data_collection[i].data:
                tmp = self._time_series_data_collection[i].data[d][ix]
                tmp_time.set_data(tmp,d)
            for l in self._time_series_data_collection[i].labels:
                tmp = self._time_series_data_collection[i].labels[l][ix]
                tmp_time.set_labels(tmp,l)               
            self._time_series_data_collection[i] = tmp_time
        return self

    def pad_time_index(self):
        timeix = []
        for i in self._time_series_data_collection:
            timeix.extend(self._time_series_data_collection[i][:][self._time_series_Ix]) 
        timeix = sorted(list(set(timeix)))
        for i in self._time_series_data_collection:
            tmp_time = Time_Series_Data()
            tmp_time.set_time_index(timeix,self._time_series_Ix)
            tmp = self._time_series_data_collection[i]
            for t in tmp.time_index:
                posList= np.isin(timeix,tmp.time_index[t])
            for d in tmp.data:
                nanList = np.empty(len(timeix))
                nanList[:] = np.nan
                nanList[posList] = tmp.data[d]
                tmp_time.set_data(nanList,d)
            for l in tmp.labels:
                nanList = np.empty(len(timeix))
                nanList[:] = np.nan
                nanList[posList] = tmp.labels[l]
                tmp_time.set_labels(nanList,l)
            self._time_series_data_collection[i] = tmp_time
        return self

    def sort(self,ascending=True,categoryList=None):
        if categoryList is None:
            categoryList = list(self._time_series_data_collection.keys())
        for i in categoryList:
            self._time_series_data_collection[i] =self._time_series_data_collection[i].sort(ascending)
        return self

    def __repr__(self):
        return str(self._time_series_data_collection)

    def __getitem__(self,ix):
        return self._time_series_data_collection[ix]

    def _expand_dict_category(self,collectionDict):
        time_series = Time_Series_Data()
        for i in collectionDict:
            tmp =collectionDict[i]
            tmp.sort()
            for t in tmp.time_index:
                time_series.set_time_index(tmp.time_index[t],t)
            for d in tmp.data:
                time_series.set_data(tmp.data[d],f'{d}_{i}')
            for l in tmp.labels:
                time_series.set_labels(tmp.labels[l],f'{l}_{i}')
            
        return {'1':time_series}

    def _expand_dict_date(self,collectionDict):
        dct = {}
        for k in collectionDict:
            tmp = {}
            a = collectionDict[k]
            for i in range(a.time_length):
                timeIx = list(a.time_index.keys())[0]
                for t in a[i]:
                    if t in a.time_index:
                        continue
                    if  not isinstance(a[i][t],list) or not isinstance(a[i][t],np.ndarray):
                        tmp[f"{t}_{a[i][timeIx]}"]=[a[i][t]]
                    else:
                        tmp[f"{t}_{a[i][timeIx]}"]=a[i][t]
            dct[k] = tmp
        return dct


    def make_dataframe(self,expandCategory,expandTimeIx):
        resDf = pd.DataFrame()
        transCollection = copy.copy(self.time_series_data_collection)
        if expandCategory:
            transCollection = self._expand_dict_category(transCollection)
        if expandTimeIx:
            transCollection = self._expand_dict_date(transCollection)
        for i in transCollection:
            if expandTimeIx == False:
                tmp = pd.DataFrame(transCollection[i][:])
            else:
                tmp = pd.DataFrame(transCollection[i])
            if expandCategory == False:
                tmp[self._categoryIx] = i
            resDf = resDf.append(tmp)
        return resDf


######## Depreciated ###########

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