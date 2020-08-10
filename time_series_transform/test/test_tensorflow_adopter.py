import pytest
from time_series_transform.transform_core_api.tensorflow_adopter import *
import os
import numpy as np


class Test_tfrecord_adopter:

    def test_single_int(self):
        data = [({'intObject':np.array([1])},1),({'intObject':np.array([-1])},1)]
        tw = TFRecord_Writer("./int.tfRecord")
        tw.write_tfRecord(data)
        tr = TFRecord_Reader('./int.tfRecord',tw.get_tfRecord_dtype())
        dataset = tr.make_tfDataset()
        assert list(dataset.as_numpy_iterator())[0]['intObject'][0] == 1.0
        os.remove('./int.tfRecord')

    def test_tensor(self):
        data = [({'tensorObject':np.array([[[1,np.nan]]])},1)]
        tw = TFRecord_Writer("./tensor.tfRecord")
        tw.write_tfRecord(data)
        tr = TFRecord_Reader('./tensor.tfRecord',tw.get_tfRecord_dtype())
        dataset = tr.make_tfDataset()
        datasetobj = list(dataset.as_numpy_iterator())[0]['tensorObject'][0]
        np.testing.assert_array_equal(datasetobj,[[1.0,np.nan]])
        os.remove('./tensor.tfRecord')

    def test_float(self):
        data = [({'floatObject':np.array([1.0])},1),({'floatObject':np.array([-1.0])},1)]
        tw = TFRecord_Writer("./float.tfRecord")
        tw.write_tfRecord(data)
        tr = TFRecord_Reader('./float.tfRecord',tw.get_tfRecord_dtype())
        dataset = tr.make_tfDataset()
        assert list(dataset.as_numpy_iterator())[0]['floatObject'][0] == 1.0
        os.remove('./float.tfRecord')