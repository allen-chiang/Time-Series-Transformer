import gc,uuid
import pandas as pd
import numpy as np
import pyarrow as pa
import tensorflow as tf
from pyarrow import parquet as pq
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from time_series_transform.base import *
from time_series_transform.sequence_transfomer import Sequence_Transformer_Base

class Time_Series_Transformer(object):

    def __init__ (self,df,dimList,encoder = LabelEncoder,encodeDict = None,seqTransformerList= None):
        """this class transfroms pandas frame into time series tensors with its corresponding categorical data
        
        the column of data frame has to be [dim1, dim2,dim....., t0,t1,t2,t....], and the index has to be the item or id
        
        Parameters
        ----------
        df : pandas data frame
            the input data frame with time series data and categorical data as columns, and item or id as index
        dimList : list or iterable
            list of categorical data in the data frame
        encoder : scikit-learn transformer like class, optional
            this attribute is used to label the categorical data, and it must implmemnt scikit-learn transformer api, by default LabelEncoder
        encodeDict : dict of label encoder object, optional
            this dictionary will use pre-trained encoder to label data --> it is designed for validation set or test set data, by default None
        seqTransformerList: list of transformer for seqential data
            this list is used for manipulating sequential data as new feature --> item of this list must implment Sequence_Transformer_Base
        """
        super().__init__()
        self._df = df
        self._dimList = dimList
        self.arr = df.drop(dimList,axis =1).values
        self.indexList = df.index.tolist()
        self._encoder = encoder
        self.labelDict,self.encodeDict = self._pandas_to_categorical_encode(encodeDict)
        self.seqTransformerList = seqTransformerList


    def _rolling_window(self,a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def _get_time_tensor(self,arr,window_size,returnY = True):
        tmp = self._rolling_window(arr,window_size+1)
        Xtensor = tmp[:,:-1].reshape(-1,window_size,1)
        Ytensor = tmp[:,-1].reshape(-1,1)
        if not returnY:
            return Xtensor
        return (Xtensor,Ytensor)

    def _tensor_transfomer(self,arr,window_size,transformer):
        if not isinstance(transformer,Sequence_Transformer_Base):
            raise ValueError('Transformer must implment Sequence_Transformer_Base')
        tmpArr = transformer.Call(arr)
        return self._get_time_tensor(tmpArr,window_size,False)

    def _tensor_factory(self,arr,window_size,categoryIx,seqTransformerList=[]):
        X,Ytensor = self._get_time_tensor(arr,window_size)
        for i in seqTransformerList:
            tmpArr = self._tensor_transfomer(arr,window_size,i)
            X = np.dstack((X,tmpArr))
        Xtensor = {}
        for i in self.labelDict:
            label = self.labelDict[i][categoryIx]
            Xtensor[i] = self._label_shape_transform(label,Ytensor.shape)
        Xtensor['time_series'] = X
        return (Xtensor,Ytensor)

    def _label_encode(self,arr,encoder):
        if encoder is None:
            encoder = self._encoder()
            enc_arr = encoder.fit_transform(arr)
        else:
            enc_arr = encoder.transform(arr)
        return enc_arr,encoder

    def _pandas_to_categorical_encode(self,encodeDict):
        if encodeDict is None:
            encodeDict = {}
        labelDict = {}
        for i in self._dimList:
            if i in encodeDict:
                enc_arr,encoder = self._label_encode(self._df[i],encodeDict[i])
            else:
                enc_arr,encoder = self._label_encode(self._df[i],None)
            encodeDict[i] = encoder
            labelDict[i] = enc_arr
        return labelDict,encodeDict

    def _label_shape_transform(self,label,shape):
        tmp = np.zeros(shape)
        tmp += label
        return tmp

    def np_to_time_tensor_generator(self,windowSize):
        """np_to_time_tensor_generator this function will prepare the df data into generator type object
        
        this function is based on _tensor_factory function to transform the data
        
        Parameters
        ----------
        windowSize : int
            window size used to group time series sequence
        
        Yields
        -------
        tuple
            it will yield (X,y)
        """
        if np.ndim(self.arr) > 1:
            for ix,v in enumerate(self.arr):
                yield self._tensor_factory(v,windowSize,ix,self.seqTransformerList)
        else:
            yield self._tensor_factory(self.arr,windowSize,0,self.seqTransformerList) 

    def get_encoder_class(self,label):
        """get_encoder_class this function will return the class number of the encoding label
        
        
        Parameters
        ----------
        label : str
            encoding class name
        
        Returns
        -------
        int
            the class number
        """
        return len(self.encodeDict[label].classes_)


class Pandas_Time_Series_Dataset(object):
    def __init__(self,pandasFrame,config={}):
        super().__init__()
        self.df = pandasFrame
        self.config = config

    def _dict_keys_values(self,data,keys):
        res = []
        for k in keys:
            res.append(data[k])
        return np.array(res)

    def _make_time_series_dataset(self,data):
        tensorDict = {}
        for i in self.config:
            process_data = self._dict_keys_values(data,self.config[i]['colNames'])
            tsf = Time_Series_Tensor_Factory(
                process_data,
                self.config[i]['tensorType']
                )
            tensor = tsf.get_time_series_tensor(
                name = i,
                **self.config[i]['param']
                )
            if self.config[i].get('sequence_stack') is not None:
                sequence_stack = self.config[i].get('sequence_stack')
                tensorDict[sequence_stack].stack_time_series_tensors(tensor)
            else:
                tensorDict[i] = tensor
        tensorList =  [ v for v in tensorDict.values() ]
        return Time_Series_Dataset(tensorList).make_dataset()

    def set_config(self,name,colNames,tensorType,param,sequence_stack,isResponseVar):
        self.config[name] = {
            'colNames':colNames,
            'tensorType':tensorType,
            'param':param,
            'sequence_stack':sequence_stack,
            'responseVariable':isResponseVar
            }

    def make_data_generator(self):
        data = self.df.to_dict('records')
        for i in data:
            res = self._make_time_series_dataset(i)
            Xtensor = {}
            Ytensor = None
            for c in self.config:
                if self.config[c].get("sequence_stack") is not None:
                    continue
                if self.config[c].get("responseVariable"):
                    Ytensor = res[c]
                else:
                    Xtensor[c] = res[c] 
            yield (Xtensor,Ytensor)
