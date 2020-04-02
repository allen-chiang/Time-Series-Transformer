import numpy as np
import pandas as pd
import tensorflow as tf

class TFRecord_Generator(object):
    def __init__(self):
        pass
    
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
        return self._bytes_feature(tf.io.serialize_tensor(value))

    def _valueDict_builder(self,data):
        valueDict = {}
        for i in data:
            if np.ndim(data[i]) > 0:
                valueDict[i] = self._tensor_feature(data[i])
            else:
                if data[i].dtype == np.int:
                    valueDict[i] = self._int64_feature(data[i])
                else:
                    valueDict[i] = self._float_feature(data[i])
        return valueDict

    def tfExample_factory(self,valueDict):
        example_proto = tf.train.Example(features=tf.train.Features(feature=valueDict))
        return example_proto.SerializeToString()

    def write_tfRecord(self,fileName,data):
        with tf.io.TFRecordWriter(fileName) as writer:
            for i in data:
                serialized_features_dataset = self.tfExample_factory(i)
                writer.write(serialized_features_dataset)

