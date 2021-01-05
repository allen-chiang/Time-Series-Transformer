import pytest
from time_series_transform.transform_core_api.tfDataset_adopter import *
import os
import numpy as np


class Test_tfrecord_adopter:

    def test_single_int(self):
        data = [{'intObject':1},{'intObject':-1}]
        tw = TFRecord_Writer("./int.tfRecord")
        tw.write_tfRecord(data)
        tr = TFRecord_Reader('./int.tfRecord',tw.get_tfRecord_dtype())
        dataset = tr.make_tfDataset()
        assert list(dataset.as_numpy_iterator())[0]['intObject'] == 1
        assert list(dataset.as_numpy_iterator())[1]['intObject'] == -1
        os.remove('./int.tfRecord')

    def test_tensor(self):
        data = [{'tensorObject':np.array([[[1,np.nan]]])}]
        tw = TFRecord_Writer("./tensor.tfRecord")
        tw.write_tfRecord(data)
        tr = TFRecord_Reader('./tensor.tfRecord',tw.get_tfRecord_dtype())
        dataset = tr.make_tfDataset()
        datasetobj = list(dataset.as_numpy_iterator())[0]['tensorObject'][0]
        np.testing.assert_array_equal(datasetobj,[[1.0,np.nan]])
        os.remove('./tensor.tfRecord')

    def test_float(self):
        data = [{'floatObject':1.0},{'floatObject':-1.0}]
        tw = TFRecord_Writer("./float.tfRecord")
        tw.write_tfRecord(data)
        tr = TFRecord_Reader('./float.tfRecord',tw.get_tfRecord_dtype())
        dataset = tr.make_tfDataset()
        assert list(dataset.as_numpy_iterator())[0]['floatObject'] == 1.0
        assert list(dataset.as_numpy_iterator())[1]['floatObject'] == -1.0
        os.remove('./float.tfRecord')


    def test_str(self):
        data = [{'strObject':'2020-13-12'},{'strObject':"asdaf!@#kj;l\n)*&dkfja"}]
        tw = TFRecord_Writer("./str.tfRecord")
        tw.write_tfRecord(data)
        tr = TFRecord_Reader('./str.tfRecord',tw.get_tfRecord_dtype())
        dataset = tr.make_tfDataset()
        print(list(dataset.as_numpy_iterator())[0])
        assert list(dataset.as_numpy_iterator())[0]['strObject'] == b'2020-13-12'
        assert list(dataset.as_numpy_iterator())[1]['strObject'] == b"asdaf!@#kj;l\n)*&dkfja"
        os.remove('./str.tfRecord')