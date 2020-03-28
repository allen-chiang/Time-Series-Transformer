import gc,uuid
import pandas as pd
import numpy as np
import pyarrow as pa
import tensorflow as tf
from pyarrow import parquet as pq
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder


class feature_engineering(object):

    def __init__ (self,df,dimList,encoder = LabelEncoder,encodeDict = None):
        super().__init__()
        self._df = df
        self._dimList = dimList
        self.arr = df.drop(dimList,axis =1).values
        self.indexList = df.index.tolist()
        self._encoder = encoder
        self.labelDict,self.encodeDict = self._pandas_to_categorical_encode(encodeDict)


    def _rolling_window(self,a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def _get_time_tensor(self,arr,window_size):
        tmp = self._rolling_window(arr,window_size+1)
        Xtensor = tmp[:,:-1].reshape(-1,window_size,1)
        Ytensor = tmp[:,-1].reshape(-1,1)
        return (Xtensor,Ytensor)

    def _tensor_factory(self,arr,window_size,categoryIx):
        X,Ytensor = self._get_time_tensor(arr,window_size)
        Xtensor = {}
        for i in self.labelDict:
            label = self.labelDict[i][categoryIx]
            Xtensor[i] = self._label_shape_transform(label,Ytensor.shape)
        Xtensor['sells'] = X
        return (Xtensor,Ytensor)

    def np_to_time_tensor_generator(self,windowSize):
        if np.ndim(self.arr) > 1:
            for ix,v in enumerate(self.arr):
                yield self._tensor_factory(v,windowSize,ix)
        else:
            yield self._tensor_factory(self.arr,windowSize,0) 

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

    def get_encoder_class(self,label):
        return len(self.encodeDict[label].classes_)
    
    def _get_tf_output_type(self):
        dct = {}
        for i in self.encodeDict:
            dct[i] = tf.int16
        dct['sells'] = tf.float32
        return (dct,tf.float32)
    
    def _get_tf_output_shape(self,window_size):
        dct = {}
        for i in self.encodeDict:
            dct[i] = tf.TensorShape([None,1])
        dct['sells'] = tf.TensorShape([None,window_size,1])
        return (dct,tf.TensorShape([None,1]))
    
    def get_tf_dataset(self,window_size):
        return tf.data.Dataset.from_generator(
                    self.np_to_time_tensor_generator,
                    self._get_tf_output_type(),
                    output_shapes = self._get_tf_output_shape(window_size),
                    args = [window_size]
        ), len(list(self.np_to_time_tensor_generator(window_size)))