import gc
import uuid
import numpy as np
import pandas as pd
import pyarrow as pa
import tensorflow as tf
from pyarrow import parquet as pq
from collections import defaultdict
from matplotlib import pyplot as plt
from time_series_transform.base import *



class Pandas_Time_Series_Dataset(object):
    def __init__(self, pandasFrame, config={}):
        """
        Pandas_Time_Series_Dataset prepared pandas data into sequence data type
        
        This class will follow the configuration to transform the pandas dataframe into sequence data
        the restriction for using this interface:
            - the column of data frame has to be [dim1, dim2,dim....., t0,t1,t2,t....], and the index has to be the item or id
            - the configuration data has to be a dictionary and follow by this template
            {
                "colName": str,
                "tensorType":{'sequence','label','category'},
                "param": {"windowSize":int,"seqSize":int,"outType":numpy datatype}
                "sequence_stack": other colName [option]
                "responseVariable": {True,False} [optional]
            }

        Parameters
        ----------
        pandasFrame : pandas DataFrame
            input data
        config : dict, optional
            the configuration to trainsform pandas dataFrame, by default {}
        """
        super().__init__()
        self.df = pandasFrame
        self.config = config

    def set_config(self, name, colNames, tensorType, sequence_stack, isResponseVar, windowSize, seqSize, outType):
        """
        set_config the setter of config
        
        this setter provide an quick entry point to setup configuration
        
        Parameters
        ----------
        name : str
            the name of the output sequence or output column
        colNames : list of string
            the name of pandas frame used for transformation
        tensorType : {'sequence','label','category'}
            provide different type of transformation
        sequence_stack : string of name for stacking
            the target name for stacking
        isResponseVar : bool
            whether the data is response variable or predictor
        windowSize: int
            sequence grouping size
        seqSize: int
            total length of sequence
        outType: numpy data type
            output data type
        """
        self.config[name] = {
            'colNames': colNames,
            'tensorType': tensorType,
            'param': {
                "windowSize": windowSize, 
                "seqSize": seqSize, 
                "outType": outType
                },
            'sequence_stack': sequence_stack,
            'responseVariable': isResponseVar
        }

    def _dict_keys_values(self, data, keys):
        res = []
        for k in keys:
            res.append(data[k])
        return np.array(res)

    def _make_time_series_dataset(self, data):
        tensorDict = {}
        for i in self.config:
            process_data = self._dict_keys_values(
                data, self.config[i]['colNames'])
            tsf = Time_Series_Tensor_Factory(
                process_data,
                self.config[i]['tensorType']
            )
            tensor = tsf.get_time_series_tensor(
                name=i,
                **self.config[i]['param']
            )
            if self.config[i].get('sequence_stack') is not None:
                sequence_stack = self.config[i].get('sequence_stack')
                tensorDict[sequence_stack].stack_time_series_tensors(tensor)
            else:
                tensorDict[i] = tensor
        tensorList = [v for v in tensorDict.values()]
        return Time_Series_Dataset(tensorList).make_dataset()

    def make_data_generator(self):
        """
        make_data_generator prepare an generator to output the transformed data
        
        
        Yields
        -------
        tuple
            it will output X data and Y data
        """
        data = self.df.to_dict('records')
        for i in data:
            res = self._make_time_series_dataset(i)
            Xtensor = {}
            Ytensor = None
            for c in self.config:
                if self.config[c].get("sequence_stack") is not None:
                    continue
                if self.config[c].get("responseVariable"):
                    Ytensor = res['data'][c]
                else:
                    Xtensor[c] = res['data'][c]
            yield (Xtensor, Ytensor)
