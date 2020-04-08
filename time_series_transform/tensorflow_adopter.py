import numpy as np
import pandas as pd
import tensorflow as tf

class TFRecord_Generator(object):
    def __init__(self,fileName,dtypeDict={},compression_type = 'GZIP'):
        self.fileName = fileName
        self._dtypeDict = dtypeDict
        self._compression_type = compression_type

    def _get_dtypes(self):
        dtypeDict = {}
        return dtypeDict
    
    def _bytes_feature(self,value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self,value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self,value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _tensor_feature(self,value):
        """Returns an bytes_list from a numpy tensor"""
        return self._bytes_feature(tf.io.serialize_tensor(value.astype(np.float32)))

    def _valueDict_builder(self,data):
        valueDict = {}
        for i in data:
            if np.ndim(data[i]) > 0:
                self._dtypeDict[i] = ('tensor',data[i].shape)
                valueDict[i] = self._tensor_feature(data[i])
            else:
                if data[i].dtype == np.int:
                    self._dtypeDict[i] = ('int',[1])
                    valueDict[i] = self._int64_feature(data[i])
                else:
                    self._dtypeDict[i] = ('float',[1])
                    valueDict[i] = self._float_feature(data[i])
        return valueDict

    def _read_tfrecord(self,serialized_example,feature_desc,dtypeDict,tensor_opt_dtype):
        record = {}
        example = tf.io.parse_single_example(serialized_example,feature_desc)
        for i in dtypeDict:
            if dtypeDict[i][0] == 'tensor':
                tmp = tf.io.parse_tensor(example[i], out_type = tensor_opt_dtype)
                tmp.set_shape(dtypeDict[i][1])
                record[i] = tmp
            else:
                record[i] = example[i]
        return record

    def tfExample_factory(self,valueDict):
        example_proto = tf.train.Example(features=tf.train.Features(feature=valueDict))
        return example_proto.SerializeToString()

    def write_tfRecord(self,data):
        with tf.io.TFRecordWriter(self.fileName,self._compression_type) as writer:
            for X,y in data:
                X['label'] = y
                valueDict = self._valueDict_builder(X)
                serialized_features_dataset = self.tfExample_factory(valueDict)
                writer.write(serialized_features_dataset)

    def feature_des_builder(self):
        feature_desc = {}
        for i in self._dtypeDict:
            if self._dtypeDict[i][0] == 'tensor':
                feature_desc[i] = tf.io.FixedLenFeature((), tf.string)
            elif self._dtypeDict[i][0] == 'float':
                feature_desc[i] = tf.io.FixedLenFeature((), tf.float32)
            elif self._dtypeDict[i][0] == 'int':
                feature_desc[i] = tf.io.FixedLenFeature((), tf.int64)
        return feature_desc

    def make_tfDataset(self,tensor_opt_dtype = tf.float32):
        feature_desc = self.feature_des_builder()
        raw_dataset = tf.data.TFRecordDataset(self.fileName,compression_type=self._compression_type)
        return raw_dataset.map(lambda x: self._read_tfrecord(x,feature_desc,self._dtypeDict,tensor_opt_dtype))

