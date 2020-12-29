import os
import copy
import pytest
import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import feather as pf
from pyarrow import parquet as pq
from time_series_transform.io.base import io_base
from time_series_transform.io.numpy import (
    from_numpy, 
    to_numpy
    )
from time_series_transform.io.pandas import (
    from_pandas, 
    to_pandas
    )
from time_series_transform.io.arrow import (
    from_arrow_record_batch, 
    from_arrow_table, 
    to_arrow_record_batch, 
    to_arrow_table
    )
from time_series_transform.transform_core_api.base import (
    Time_Series_Data, 
    Time_Series_Data_Collection
    )
from time_series_transform.io.parquet import (
    from_parquet,
    to_parquet
    )
from time_series_transform.io.feather import (
    from_feather,
    to_feather
    )

@pytest.fixture(scope = 'class')
def dictList_single():
    return {
        'time': [1, 2],
        'data': [1, 2]
    }

@pytest.fixture(scope = 'class')
def dictList_collection():
    return {
        'time': [1,2,1,3],
        'data':[1,2,1,2],
        'category':[1,1,2,2]
    }

@pytest.fixture(scope = 'class')
def expect_single_expandTime():
    return {
        'data_1':[1],
        'data_2':[2]
    }

@pytest.fixture(scope = 'class')
def expect_single_seperateLabel():
    return [{
        'time': [1, 2],
        'data': [1, 2]
    },
    {
        'data_label': [1, 2]
    }]

@pytest.fixture(scope = 'class')
def expect_collection_seperateLabel():
    return [{
        'time': [1,2,1,3],
        'data':[1,2,1,2],
        'category':[1,1,2,2]
    },
    {
        'data_label':[1,2,1,2]
    }
]

@pytest.fixture(scope = 'class')
def expect_collection_expandTime():
    return {
        'pad': {
            'data_1':[1,1],
            'data_2':[2,np.nan],
            'data_3':[np.nan,2],
            'category':[1,2]
        },
        'remove': {
            'data_1':[1,1],
            'category':[1,2]            
        }
    }
@pytest.fixture(scope = 'class')
def expect_collection_expandCategory():
    return {
        'pad': {
            'time':[1,2,3],
            'data_1':[1,2,np.nan],
            'data_2':[1,np.nan,2]
        },
        'remove': {
            'time':[1],
            'data_1':[1],
            'data_2':[1]            
        }        
    }

@pytest.fixture(scope = 'class')
def expect_collection_expandFull():
    return {
        'pad': {
            'data_1_1':[1],
            'data_2_1':[1],
            'data_1_2':[2],
            'data_2_2':[np.nan],
            'data_1_3':[np.nan],
            'data_2_3':[2]
        },
        'remove': {
            'data_1_1':[1],
            'data_2_1':[1], 
        }        
    }

@pytest.fixture(scope = 'class')
def expect_collection_noExpand():
    return {
        'ignore':{
        'time': [1,2,1,3],
        'data':[1,2,1,2],
        'category':[1,1,2,2]
        },
        'pad': {
            'time': [1,2,3,1,2,3],
            'data':[1,2,np.nan,1,np.nan,2],
            'category':[1,1,1,2,2,2]
        },
        'remove': {
            'time': [1,1],
            'data':[1,1],
            'category':[1,2]
        }        
    }

@pytest.fixture(scope = 'class')
def seq_single():
    return {
        'time':[1,2,3],
        'data':[[1,2,3],[11,12,13],[21,22,23]]
    }

@pytest.fixture(scope = 'class')
def seq_collection():
    return {
        'time':[1,2,1,2],
        'data':[[1,2],[1,2],[2,2],[2,2]],
        'category':[1,1,2,2]
    }

@pytest.fixture(scope = 'class')
def expect_seq_collection():
    return {
        'data_1_1':[[1,2]],
        'data_2_1':[[2,2]],
        'data_1_2':[[1,2]],
        'data_2_2':[[2,2]]
    }



class Test_base_io:

    def test_base_io_from_single(self, dictList_single,expect_single_expandTime):
        ExpandTimeAns = expect_single_expandTime
        data = dictList_single
        ts = Time_Series_Data()
        ts = ts.set_time_index(data['time'], 'time')
        ts = ts.set_data(data['data'], 'data')
        io = io_base(ts, 'time', None)
        timeSeries = io.from_single(False)
        for i in timeSeries:
            assert timeSeries[i].tolist() == data[i]
        timeSeries = io.from_single(True)
        for i in timeSeries:
            assert timeSeries[i] == ExpandTimeAns[i]

    def test_base_io_to_single(self, dictList_single):
        data = dictList_single
        ts = Time_Series_Data()
        ts = ts.set_time_index(data['time'], 'time')
        ts = ts.set_data(data['data'], 'data')
        io = io_base(data, 'time', None)
        assert io.to_single() == ts

    def test_base_io_from_collection_expandTime(self, dictList_collection,expect_collection_expandTime):
        noChange = dictList_collection
        expand = expect_collection_expandTime
        fullExpand = []
        data = dictList_collection
        ts = Time_Series_Data()
        ts = ts.set_time_index(data['time'], 'time')
        ts = ts.set_data(data['data'], 'data')
        ts = ts.set_data(data['category'],'category')
        tsc = Time_Series_Data_Collection(ts,'time','category')
        io = io_base(tsc, 'time', 'category')
        with pytest.raises(ValueError):
            timeSeries = io.from_collection(False,True,'ignore')
        timeSeries = io.from_collection(False,True,'pad')
        for i in timeSeries:
            np.testing.assert_equal(timeSeries[i],expand['pad'][i])
        timeSeries = io.from_collection(False,True,'remove')
        for i in timeSeries:
            np.testing.assert_equal(timeSeries[i],expand['remove'][i])

    def test_base_io_from_collection_expandCategory(self, dictList_collection,expect_collection_expandCategory):
        noChange = dictList_collection
        expand = expect_collection_expandCategory
        fullExpand = []
        data = dictList_collection
        ts = Time_Series_Data()
        ts = ts.set_time_index(data['time'], 'time')
        ts = ts.set_data(data['data'], 'data')
        ts = ts.set_data(data['category'],'category')
        tsc = Time_Series_Data_Collection(ts,'time','category')
        io = io_base(tsc, 'time', 'category')
        with pytest.raises(ValueError):
            timeSeries = io.from_collection(True,False,'ignore')
        timeSeries = io.from_collection(True,False,'pad')
        for i in timeSeries:
            np.testing.assert_equal(timeSeries[i],expand['pad'][i])
        timeSeries = io.from_collection(True,False,'remove')
        for i in timeSeries:
            np.testing.assert_equal(timeSeries[i],expand['remove'][i])

    def test_base_io_from_collection_expandFull(self, dictList_collection,expect_collection_expandFull):
        noChange = dictList_collection
        expand = expect_collection_expandFull
        fullExpand = []
        data = dictList_collection
        ts = Time_Series_Data()
        ts = ts.set_time_index(data['time'], 'time')
        ts = ts.set_data(data['data'], 'data')
        ts = ts.set_data(data['category'],'category')
        tsc = Time_Series_Data_Collection(ts,'time','category')
        io = io_base(tsc, 'time', 'category')
        timeSeries = io.from_collection(True,True,'pad')
        for i in timeSeries:
           np.testing.assert_equal(timeSeries[i],expand['pad'][i])
        timeSeries = io.from_collection(True,True,'remove')
        for i in timeSeries:
            np.testing.assert_equal(timeSeries[i],expand['remove'][i])

    def test_base_io_to_collection(self, dictList_collection):
        dataList = dictList_collection
        io = io_base(dataList, 'time', 'category')
        testData = io.to_collection()
        tsd = Time_Series_Data(dataList,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        assert  testData== tsc

    def test_base_io_from_collection_no_expand(self,dictList_collection,expect_collection_noExpand):
        noChange = dictList_collection
        expand = expect_collection_noExpand
        data = dictList_collection
        ts = Time_Series_Data()
        ts = ts.set_time_index(data['time'], 'time')
        ts = ts.set_data(data['data'], 'data')
        ts = ts.set_data(data['category'],'category')
        tsc = Time_Series_Data_Collection(ts,'time','category')
        io = io_base(tsc, 'time', 'category')
        timeSeries = io.from_collection(False,False,'ignore')
        for i in timeSeries:
            np.testing.assert_array_equal(timeSeries[i],expand['ignore'][i])
        timeSeries = io.from_collection(False,False,'pad')
        for i in timeSeries:
            np.testing.assert_equal(timeSeries[i],expand['pad'][i])
        timeSeries = io.from_collection(False,False,'remove')
        for i in timeSeries:
            np.testing.assert_equal(timeSeries[i],expand['remove'][i])

class Test_Pandas_IO:
    def test_from_pandas_single(self,dictList_single):
        data = dictList_single
        df = pd.DataFrame(dictList_single)
        tsd = Time_Series_Data(data,'time')
        testData = from_pandas(df,'time',None)
        assert tsd == testData

    def test_from_pandas_collection(self,dictList_collection):
        data = dictList_collection
        df = pd.DataFrame(dictList_collection)
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        testData = from_pandas(df,'time','category')
        assert tsc == testData

    def test_to_pandas_single(self,dictList_single,expect_single_expandTime):
        data = dictList_single
        df = pd.DataFrame(data)
        expandTime = pd.DataFrame(expect_single_expandTime)
        tsd = Time_Series_Data(data,'time')
        testData = to_pandas(
            tsd,
            expandCategory= None,
            expandTime=False,
            preprocessType= None
            )
        pd.testing.assert_frame_equal(testData,df,check_dtype=False)
        testData = to_pandas(
            tsd,
            expandCategory= None,
            expandTime=True,
            preprocessType= None
            )
        pd.testing.assert_frame_equal(testData,expandTime,check_dtype=False)

    def test_to_pandas_collection_expandTime(self,dictList_collection,expect_collection_expandTime):
        data = dictList_collection
        expandTime_pad = pd.DataFrame(expect_collection_expandTime['pad'])
        expandTime_remove = pd.DataFrame(expect_collection_expandTime['remove'])
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        testData = to_pandas(
            tsc,
            expandCategory= False,
            expandTime=True,
            preprocessType= 'pad'
            )
        pd.testing.assert_frame_equal(testData,expandTime_pad,check_dtype=False)
        testData = to_pandas(
            tsc,
            expandCategory= False,
            expandTime=True,
            preprocessType= 'remove'
            )
        pd.testing.assert_frame_equal(testData,expandTime_remove,check_dtype=False)
        with pytest.raises(ValueError):
            timeSeries = to_pandas(tsc,False,True,'ignore')        


    def test_to_pandas_collection_expandCategory(self,dictList_collection,expect_collection_expandCategory):
        data = dictList_collection
        expandTime_pad = pd.DataFrame(expect_collection_expandCategory['pad'])
        expandTime_remove = pd.DataFrame(expect_collection_expandCategory['remove'])
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        testData = to_pandas(
            tsc,
            expandCategory= True,
            expandTime=False,
            preprocessType= 'pad'
            )
        pd.testing.assert_frame_equal(testData,expandTime_pad,check_dtype=False)
        testData = to_pandas(
            tsc,
            expandCategory= True,
            expandTime=False,
            preprocessType= 'remove'
            )
        pd.testing.assert_frame_equal(testData,expandTime_remove,check_dtype=False)
        with pytest.raises(ValueError):
            timeSeries = to_pandas(tsc,True,False,'ignore')    

    def test_to_pandas_collection_expandFull(self,dictList_collection,expect_collection_expandFull):
        data = dictList_collection
        expandTime_pad = pd.DataFrame(expect_collection_expandFull['pad'])
        expandTime_remove = pd.DataFrame(expect_collection_expandFull['remove'])
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        testData = to_pandas(
            tsc,
            expandCategory= True,
            expandTime=True,
            preprocessType= 'pad'
            )
        pd.testing.assert_frame_equal(testData,expandTime_pad,check_dtype=False)
        testData = to_pandas(
            tsc,
            expandCategory= True,
            expandTime=True,
            preprocessType= 'remove'
            )
        pd.testing.assert_frame_equal(testData,expandTime_remove,check_dtype=False)

    def test_to_pandas_collection_noExpand(self,dictList_collection,expect_collection_noExpand):
        data = dictList_collection
        expandTime_ignore = pd.DataFrame(expect_collection_noExpand['ignore'])
        expandTime_pad = pd.DataFrame(expect_collection_noExpand['pad'])
        expandTime_remove = pd.DataFrame(expect_collection_noExpand['remove'])
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        testData = to_pandas(
            tsc,
            expandCategory= False,
            expandTime=False,
            preprocessType= 'pad'
            )
        pd.testing.assert_frame_equal(testData,expandTime_pad,check_dtype=False)
        testData = to_pandas(
            tsc,
            expandCategory= False,
            expandTime=False,
            preprocessType= 'remove'
            )
        pd.testing.assert_frame_equal(testData,expandTime_remove,check_dtype=False)
        testData = to_pandas(
            tsc,
            expandCategory= False,
            expandTime=False,
            preprocessType= 'ignore'
            )
        pd.testing.assert_frame_equal(testData,expandTime_ignore,check_dtype=False)

    def test_to_pandas_seperateLabels_single(self,dictList_single,expect_single_seperateLabel):
        data = dictList_single
        expectedX, expectedY = expect_single_seperateLabel
        expectedX = pd.DataFrame(expectedX)
        expectedY = pd.DataFrame(expectedY)
        tsd = Time_Series_Data(data,'time')
        tsd.set_labels([1,2],'data_label')
        x, y  = to_pandas(tsd,False,False,'ignore',True)
        print(x)
        print(y)
        pd.testing.assert_frame_equal(x,expectedX,check_dtype=False)
        pd.testing.assert_frame_equal(y,expectedY,check_dtype=False)

    def test_to_pandas_seperateLabels_collection(self,dictList_collection,expect_collection_seperateLabel):
        data = dictList_collection
        expectedX, expectedY = expect_collection_seperateLabel
        expectedX = pd.DataFrame(expectedX)
        expectedY = pd.DataFrame(expectedY)
        tsd = Time_Series_Data(data,'time')
        tsd = tsd.set_labels([1,2,1,2],'data_label')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        x, y  = to_pandas(tsc,False,False,'ignore',True)
        print(y)
        pd.testing.assert_frame_equal(x,expectedX,check_dtype=False)
        pd.testing.assert_frame_equal(y,expectedY,check_dtype=False) 

    def test_to_pandas_single_sequence(self,seq_single):
        data = seq_single
        df= pd.DataFrame(data)
        tsd = Time_Series_Data(data,'time')
        test = to_pandas(tsd,False,False,'ignore',False)
        pd.testing.assert_frame_equal(test,df,False)

    def test_to_pandas_collection_sequence(self,seq_collection,expect_seq_collection):
        data = seq_collection
        df = pd.DataFrame(data)
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        test = to_pandas(tsc,False,False,'ignore')
        pd.testing.assert_frame_equal(df,test,False)
        test = to_pandas(tsc,True,True,'ignore')
        full = pd.DataFrame(expect_seq_collection)
        print(test)
        print(full)
        test = test.reindex(sorted(df.columns), axis=1)
        full = full.reindex(sorted(df.columns), axis=1)
        pd.testing.assert_frame_equal(test,full,False)

class Test_Numpy_IO:
    def test_from_numpy_single(self,dictList_single):
        data = dictList_single
        tsd = Time_Series_Data()
        tsd.set_time_index(data['time'],0)
        tsd.set_data(data['data'],1)
        numpydata = pd.DataFrame(dictList_single).values
        testData = from_numpy(numpydata,0,None)
        assert tsd == testData

    def test_from_numpy_collection(self,dictList_collection):
        data = dictList_collection
        numpyData = pd.DataFrame(data).values
        numpyDataDict = pd.DataFrame(pd.DataFrame(data).values).to_dict('list')
        testData = from_numpy(numpyData,0,2)
        tsd = Time_Series_Data(numpyDataDict,0)
        assert testData == Time_Series_Data_Collection(tsd,0,2)

    def test_to_numpy_single(self,dictList_single,expect_single_expandTime):
        data = dictList_single
        numpyData = pd.DataFrame(data).values
        expandTime = pd.DataFrame(expect_single_expandTime).values
        tsd = Time_Series_Data()
        tsd.set_time_index(data['time'],0)
        tsd.set_data(data['data'],1)
        testData = to_numpy(
            tsd,
            expandCategory= None,
            expandTime=False,
            preprocessType= None
            )
        np.testing.assert_equal(testData,numpyData)
        testData = to_numpy(
            tsd,
            expandCategory= None,
            expandTime=True,
            preprocessType= None
            )
        np.testing.assert_equal(testData,expandTime)

    def test_to_numpy_collection_expandTime(self,dictList_collection,expect_collection_expandTime):
        data = dictList_collection
        results = expect_collection_expandTime
        numpyData = pd.DataFrame(data).values
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        pad_numpy = to_numpy(
            tsc,
            expandCategory = False,
            expandTime = True,
            preprocessType='pad'
            )
        remove_numpy = to_numpy(
            tsc,
            expandCategory = False,
            expandTime = True,
            preprocessType='remove'
            )
        np.testing.assert_equal(pad_numpy,pd.DataFrame(results['pad']).values)
        np.testing.assert_equal(remove_numpy,pd.DataFrame(results['remove']).values)
        with pytest.raises(ValueError):
            timeSeries = to_numpy(tsc,False,True,'ignore')

    def test_to_numpy_collection_expandCategory(self,dictList_collection,expect_collection_expandCategory):
        data = dictList_collection
        results = expect_collection_expandCategory
        numpyData = pd.DataFrame(data).values
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        pad_numpy = to_numpy(
            tsc,
            expandCategory = True,
            expandTime = False,
            preprocessType='pad'
            )
        remove_numpy = to_numpy(
            tsc,
            expandCategory = True,
            expandTime = False,
            preprocessType='remove'
            )
        np.testing.assert_equal(pad_numpy,pd.DataFrame(results['pad']).values)
        np.testing.assert_equal(remove_numpy,pd.DataFrame(results['remove']).values)
        with pytest.raises(ValueError):
            timeSeries = to_numpy(tsc,False,True,'ignore')

    def test_to_numpy_collection_expandFull(self,dictList_collection,expect_collection_expandFull):
        data = dictList_collection
        results = expect_collection_expandFull
        numpyData = pd.DataFrame(data).values
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        pad_numpy = to_numpy(
            tsc,
            expandCategory = True,
            expandTime = True,
            preprocessType='pad'
            )
        np.testing.assert_equal(pad_numpy,pd.DataFrame(results['pad']).values)
        remove_numpy = to_numpy(
            tsc,
            expandCategory = True,
            expandTime = True,
            preprocessType='remove'
            )
        np.testing.assert_equal(remove_numpy,pd.DataFrame(results['remove']).values)

    def test_to_numpy_collection_noExpand(self,dictList_collection,expect_collection_noExpand):
        data = dictList_collection
        results = expect_collection_noExpand
        numpyData = pd.DataFrame(data).values
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        pad_numpy = to_numpy(
            tsc,
            expandCategory = False,
            expandTime = False,
            preprocessType='pad'
            )
        np.testing.assert_equal(pad_numpy,pd.DataFrame(results['pad']).values)
        remove_numpy = to_numpy(
            tsc,
            expandCategory = False,
            expandTime = False,
            preprocessType='remove'
            )
        np.testing.assert_equal(remove_numpy,pd.DataFrame(results['remove']).values)
        ignore_numpy = to_numpy(
            tsc,
            expandCategory = False,
            expandTime = False,
            preprocessType='ignore'
            )
        np.testing.assert_equal(ignore_numpy,pd.DataFrame(results['ignore']).values)

    def test_to_numpy_seperateLabel_single(self,dictList_single,expect_single_seperateLabel):
        data = dictList_single
        expectedX, expectedY = expect_single_seperateLabel
        expectedX = pd.DataFrame(expectedX).values
        expectedY = pd.DataFrame(expectedY).values
        tsd = Time_Series_Data(data,'time')
        tsd.set_labels([1,2],'data_label')
        x, y  = to_numpy(tsd,False,False,'ignore',True)
        print(x)
        print(y)
        np.testing.assert_equal(x,expectedX)
        np.testing.assert_equal(y,expectedY)

    def test_to_numpy_seperateLabel_collection(self,dictList_collection,expect_collection_seperateLabel):
        data = dictList_collection
        expectedX, expectedY = expect_collection_seperateLabel
        expectedX = pd.DataFrame(expectedX).values
        expectedY = pd.DataFrame(expectedY).values
        tsd = Time_Series_Data(data,'time')
        tsd = tsd.set_labels([1,2,1,2],'data_label')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        x, y  = to_numpy(tsc,False,False,'ignore',True)
        np.testing.assert_equal(x,expectedX)
        np.testing.assert_equal(y,expectedY)

    def test_to_numpy_single_sequence(self,seq_single):
        data = seq_single
        df= pd.DataFrame(data).values
        tsd = Time_Series_Data(data,'time')
        test = to_numpy(tsd,False,False,'ignore',False)
        np.testing.assert_equal(df,test)

    def test_to_numpy_collection_sequence(self,seq_collection,expect_seq_collection):
        data = seq_collection
        df = pd.DataFrame(data).values
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        test = to_numpy(tsc,False,False,'ignore')
        for i in range(len(test)):
            if isinstance(test[i][1],np.ndarray):
                test[i][1] = test[i][1].tolist()
        np.testing.assert_equal(df,test)
        test = to_numpy(tsc,True,True,'ignore')
        full = pd.DataFrame(expect_seq_collection).values
        for i in range(len(test[0])):
            if isinstance(test[0][i],np.ndarray):
                test[0][i] = test[0][i].tolist()
        np.testing.assert_equal(full,test)
        

class Test_Arrow_IO:
    def test_from_arrow_table_single(self,dictList_single):
        data = dictList_single
        df = pd.DataFrame(dictList_single)
        table = pa.Table.from_pandas(df)
        testData = from_arrow_table(table,'time',None)
        tsd = Time_Series_Data(data,'time')
        assert tsd == testData

    def test_from_arrow_table_collection(self,dictList_collection):
        data = dictList_collection
        df = pd.DataFrame(dictList_collection)
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        table = pa.Table.from_pandas(df)
        testData = from_arrow_table(table,'time','category')
        assert tsc == testData
    
    def test_from_arrow_batch_single(self,dictList_single):
        data = dictList_single
        df = pd.DataFrame(dictList_single)
        table = pa.RecordBatch.from_pandas(df,preserve_index = False)
        testData = from_arrow_record_batch(table,'time',None)
        tsd = Time_Series_Data(data,'time')
        assert tsd == testData

    def test_from_arrow_batch_collection(self,dictList_collection):
        data = dictList_collection
        df = pd.DataFrame(dictList_collection)
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        table = pa.RecordBatch.from_pandas(df,preserve_index = False)
        testData = from_arrow_record_batch(table,'time','category')
        assert tsc == testData

    def test_to_arrow_table_single(self,dictList_single,expect_single_expandTime):
        data = dictList_single
        df = pd.DataFrame(data)
        expandTime = pd.DataFrame(expect_single_expandTime)
        tsd = Time_Series_Data(data,'time')
        testData = to_arrow_table(
            tsd,
            expandCategory= None,
            expandTime=False,
            preprocessType= None
            ).to_pandas()
        pd.testing.assert_frame_equal(testData,df,check_dtype=False)
        testData = to_arrow_table(
            tsd,
            expandCategory= None,
            expandTime=True,
            preprocessType= None
            ).to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime,check_dtype=False)

    def test_to_arrow_table_collection_expandTime(self,dictList_collection,expect_collection_expandTime):
        data = dictList_collection
        expandTime_pad = pd.DataFrame(expect_collection_expandTime['pad'])
        expandTime_remove = pd.DataFrame(expect_collection_expandTime['remove'])
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        testData = to_arrow_table(
            tsc,
            expandCategory= False,
            expandTime=True,
            preprocessType= 'pad'
            ).to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_pad,check_dtype=False)
        testData = to_arrow_table(
            tsc,
            expandCategory= False,
            expandTime=True,
            preprocessType= 'remove'
            ).to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_remove,check_dtype=False)
        with pytest.raises(ValueError):
            tsc = Time_Series_Data_Collection(tsd,'time','category')
            timeSeries = to_pandas(tsc,False,True,'ignore')        


    def test_to_arrow_table_collection_expandCategory(self,dictList_collection,expect_collection_expandCategory):
        data = dictList_collection
        expandTime_pad = pd.DataFrame(expect_collection_expandCategory['pad'])
        expandTime_remove = pd.DataFrame(expect_collection_expandCategory['remove'])
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        testData = to_arrow_table(
            tsc,
            expandCategory= True,
            expandTime=False,
            preprocessType= 'pad'
            ).to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_pad,check_dtype=False)
        testData = to_arrow_table(
            tsc,
            expandCategory= True,
            expandTime=False,
            preprocessType= 'remove'
            ).to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_remove,check_dtype=False)
        with pytest.raises(ValueError):
            timeSeries = to_pandas(tsc,True,False,'ignore')    

    def test_to_arrow_table_collection_expandFull(self,dictList_collection,expect_collection_expandFull):
        data = dictList_collection
        expandTime_pad = pd.DataFrame(expect_collection_expandFull['pad'])
        expandTime_remove = pd.DataFrame(expect_collection_expandFull['remove'])
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        testData = to_arrow_table(
            tsc,
            expandCategory= True,
            expandTime=True,
            preprocessType= 'pad'
            ).to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_pad,check_dtype=False)
        testData = to_arrow_table(
            tsc,
            expandCategory= True,
            expandTime=True,
            preprocessType= 'remove'
            ).to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_remove,check_dtype=False)

    def test_to_arrow_table_collection_noExpand(self,dictList_collection,expect_collection_noExpand):
        data = dictList_collection
        expandTime_ignore = pd.DataFrame(expect_collection_noExpand['ignore'])
        expandTime_pad = pd.DataFrame(expect_collection_noExpand['pad'])
        expandTime_remove = pd.DataFrame(expect_collection_noExpand['remove'])
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        testData = to_arrow_table(
            tsc,
            expandCategory= False,
            expandTime=False,
            preprocessType= 'pad'
            ).to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_pad,check_dtype=False)
        testData = to_arrow_table(
            tsc,
            expandCategory= False,
            expandTime=False,
            preprocessType= 'remove'
            ).to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_remove,check_dtype=False)
        testData = to_arrow_table(
            tsc,
            expandCategory= False,
            expandTime=False,
            preprocessType= 'ignore'
            ).to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_ignore,check_dtype=False)

    def test_to_arrow_table_seperateLabels_single(self,dictList_single,expect_single_seperateLabel):
        data = dictList_single
        expectedX, expectedY = expect_single_seperateLabel
        expectedX = pd.DataFrame(expectedX)
        expectedY = pd.DataFrame(expectedY)
        tsd = Time_Series_Data(data,'time')
        tsd.set_labels([1,2],'data_label')
        x, y  = to_arrow_table(tsd,False,False,'ignore',True)
        x = x.to_pandas()
        y = y.to_pandas()
        print(x)
        print(y)
        pd.testing.assert_frame_equal(x,expectedX,check_dtype=False)
        pd.testing.assert_frame_equal(y,expectedY,check_dtype=False)

    def test_to_arrow_table_seperateLabels_collection(self,dictList_collection,expect_collection_seperateLabel):
        data = dictList_collection
        expectedX, expectedY = expect_collection_seperateLabel
        expectedX = pd.DataFrame(expectedX)
        expectedY = pd.DataFrame(expectedY)
        tsd = Time_Series_Data(data,'time')
        tsd = tsd.set_labels([1,2,1,2],'data_label')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        x, y  = to_arrow_table(tsc,False,False,'ignore',True)
        x = x.to_pandas()
        y = y.to_pandas()
        print(y)
        pd.testing.assert_frame_equal(x,expectedX,check_dtype=False)
        pd.testing.assert_frame_equal(y,expectedY,check_dtype=False)

    def test_to_arrow_table_single_sequence(self,seq_single):
        data = seq_single
        df= pd.DataFrame(data)
        tsd = Time_Series_Data(data,'time')
        test = to_arrow_table(tsd,False,False,'ignore',False).to_pandas()
        pd.testing.assert_frame_equal(test,df,False)

    def test_to_arrow_table_collection_sequence(self,seq_collection,expect_seq_collection):
        data = seq_collection
        df = pd.DataFrame(data)
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        test = to_arrow_table(tsc,False,False,'ignore').to_pandas()
        pd.testing.assert_frame_equal(df,test,False)
        test = to_arrow_table(tsc,True,True,'ignore').to_pandas()
        full = pd.DataFrame(expect_seq_collection)
        print(test)
        print(full)
        test = test.reindex(sorted(df.columns), axis=1)
        full = full.reindex(sorted(df.columns), axis=1)
        pd.testing.assert_frame_equal(test,full,False)

###

    def record_batch_to_pandas(self,batchList):
        df = None
        for i in batchList:
            if df is None:
                df = i.to_pandas()
                continue
            df = df.append(i.to_pandas(),ignore_index = True)
        return df

    def test_to_arrow_batch_single(self,dictList_single,expect_single_expandTime):
        data = dictList_single
        df = pd.DataFrame(data)
        expandTime = pd.DataFrame(expect_single_expandTime)
        tsd = Time_Series_Data(data,'time')
        testData = to_arrow_record_batch(
            tsd,
            expandCategory= None,
            expandTime=False,
            preprocessType= None,
            max_chunksize = 1
            )
        testData = self.record_batch_to_pandas(testData)
        pd.testing.assert_frame_equal(testData,df,check_dtype=False)
        testData = to_arrow_record_batch(
            tsd,
            expandCategory= None,
            expandTime=True,
            preprocessType= None,
            max_chunksize = 1
            )
        testData = self.record_batch_to_pandas(testData)
        pd.testing.assert_frame_equal(testData,expandTime,check_dtype=False)

    def test_to_arrow_batch_collection_expandTime(self,dictList_collection,expect_collection_expandTime):
        data = dictList_collection
        expandTime_pad = pd.DataFrame(expect_collection_expandTime['pad'])
        expandTime_remove = pd.DataFrame(expect_collection_expandTime['remove'])
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        testData = to_arrow_record_batch(
            tsc,
            expandCategory= False,
            expandTime=True,
            preprocessType= 'pad',
            max_chunksize=1
            )
        testData = self.record_batch_to_pandas(testData)
        pd.testing.assert_frame_equal(testData,expandTime_pad,check_dtype=False)
        testData = to_arrow_record_batch(
            tsc,
            expandCategory= False,
            expandTime=True,
            preprocessType= 'remove',
            max_chunksize=1
            )
        testData = self.record_batch_to_pandas(testData)
        pd.testing.assert_frame_equal(testData,expandTime_remove,check_dtype=False)
        with pytest.raises(ValueError):
            timeSeries = to_pandas(tsc,False,True,'ignore')        


    def test_to_arrow_batch_collection_expandCategory(self,dictList_collection,expect_collection_expandCategory):
        data = dictList_collection
        expandTime_pad = pd.DataFrame(expect_collection_expandCategory['pad'])
        expandTime_remove = pd.DataFrame(expect_collection_expandCategory['remove'])
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        testData = to_arrow_record_batch(
            tsc,
            expandCategory= True,
            expandTime=False,
            preprocessType= 'pad',
            max_chunksize=1
            )
        testData = self.record_batch_to_pandas(testData)
        pd.testing.assert_frame_equal(testData,expandTime_pad,check_dtype=False)
        testData = to_arrow_record_batch(
            tsc,
            expandCategory= True,
            expandTime=False,
            preprocessType= 'remove',
            max_chunksize=1
            )
        testData = self.record_batch_to_pandas(testData)
        pd.testing.assert_frame_equal(testData,expandTime_remove,check_dtype=False)
        with pytest.raises(ValueError):
            timeSeries = to_pandas(tsc,True,False,'ignore')    

    def test_to_arrow_batch_collection_expandFull(self,dictList_collection,expect_collection_expandFull):
        data = dictList_collection
        expandTime_pad = pd.DataFrame(expect_collection_expandFull['pad'])
        expandTime_remove = pd.DataFrame(expect_collection_expandFull['remove'])
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        testData = to_arrow_record_batch(
            tsc,
            expandCategory= True,
            expandTime=True,
            preprocessType= 'pad',
            max_chunksize=1
            )
        testData = self.record_batch_to_pandas(testData)
        pd.testing.assert_frame_equal(testData,expandTime_pad,check_dtype=False)
        testData = to_arrow_record_batch(
            tsc,
            expandCategory= True,
            expandTime=True,
            preprocessType= 'remove',
            max_chunksize=1
            )
        testData = self.record_batch_to_pandas(testData)
        pd.testing.assert_frame_equal(testData,expandTime_remove,check_dtype=False)

    def test_to_arrow_batch_collection_noExpand(self,dictList_collection,expect_collection_noExpand):
        data = dictList_collection
        expandTime_ignore = pd.DataFrame(expect_collection_noExpand['ignore'])
        expandTime_pad = pd.DataFrame(expect_collection_noExpand['pad'])
        expandTime_remove = pd.DataFrame(expect_collection_noExpand['remove'])
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        testData = to_arrow_record_batch(
            tsc,
            expandCategory= False,
            expandTime=False,
            preprocessType= 'pad',
            max_chunksize=1
            )
        testData = self.record_batch_to_pandas(testData)
        pd.testing.assert_frame_equal(testData,expandTime_pad,check_dtype=False)
        testData = to_arrow_record_batch(
            tsc,
            expandCategory= False,
            expandTime=False,
            preprocessType= 'remove',
            max_chunksize=1
            )
        testData = self.record_batch_to_pandas(testData)
        pd.testing.assert_frame_equal(testData,expandTime_remove,check_dtype=False)
        testData = to_arrow_record_batch(
            tsc,
            expandCategory= False,
            expandTime=False,
            preprocessType= 'ignore',
            max_chunksize=1
            )
        testData = self.record_batch_to_pandas(testData)
        pd.testing.assert_frame_equal(testData,expandTime_ignore,check_dtype=False)

    def test_to_arrow_batch_seperateLabels_single(self,dictList_single,expect_single_seperateLabel):
        data = dictList_single
        expectedX, expectedY = expect_single_seperateLabel
        expectedX = pd.DataFrame(expectedX)
        expectedY = pd.DataFrame(expectedY)
        tsd = Time_Series_Data(data,'time')
        tsd.set_labels([1,2],'data_label')
        x, y  = to_arrow_record_batch(tsd,1,False,False,'ignore',True)
        x = self.record_batch_to_pandas(x)
        y = self.record_batch_to_pandas(y)
        print(x)
        print(y)
        pd.testing.assert_frame_equal(x,expectedX,check_dtype=False)
        pd.testing.assert_frame_equal(y,expectedY,check_dtype=False)

    def test_to_arrow_table_seperateLabels_collection(self,dictList_collection,expect_collection_seperateLabel):
        data = dictList_collection
        expectedX, expectedY = expect_collection_seperateLabel
        expectedX = pd.DataFrame(expectedX)
        expectedY = pd.DataFrame(expectedY)
        tsd = Time_Series_Data(data,'time')
        tsd = tsd.set_labels([1,2,1,2],'data_label')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        x, y  = to_arrow_record_batch(tsc,1,False,False,'ignore',True)
        x = self.record_batch_to_pandas(x)
        y = self.record_batch_to_pandas(y)
        print(y)
        pd.testing.assert_frame_equal(x,expectedX,check_dtype=False)
        pd.testing.assert_frame_equal(y,expectedY,check_dtype=False)


class Test_Parquet_IO:
    def test_from_parquet_single(self,dictList_single):
        data = dictList_single
        df = pd.DataFrame(dictList_single)
        table = pa.Table.from_pandas(df)
        pq.write_table(table,'test.parquet')
        testData = from_parquet('test.parquet','time',None)
        tsd = Time_Series_Data(data,'time')
        assert tsd == testData
        os.remove('test.parquet')

    def test_from_parquet_collection(self,dictList_collection):
        data = dictList_collection
        df = pd.DataFrame(dictList_collection)
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        table = pa.Table.from_pandas(df)
        pq.write_table(table,'test_collection.parquet')
        testData = from_parquet('test_collection.parquet','time','category')
        assert tsc == testData
        os.remove('test_collection.parquet')

###########
    def test_to_parquet_single(self,dictList_single,expect_single_expandTime):
        data = dictList_single
        df = pd.DataFrame(data)
        expandTime = pd.DataFrame(expect_single_expandTime)
        tsd = Time_Series_Data(data,'time')
        to_parquet(
            'test.parquet',
            tsd,
            expandCategory= None,
            expandTime=False,
            preprocessType= None
            )
        testData = pq.read_table('test.parquet').to_pandas()
        pd.testing.assert_frame_equal(testData,df,check_dtype=False)
        to_parquet(
            'test.parquet',
            tsd,
            expandCategory= None,
            expandTime=True,
            preprocessType= None
            )
        testData = pq.read_table('test.parquet').to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime,check_dtype=False)
        os.remove('test.parquet')

    def test_to_parquet_collection_expandTime(self,dictList_collection,expect_collection_expandTime):
        data = dictList_collection
        expandTime_pad = pd.DataFrame(expect_collection_expandTime['pad'])
        expandTime_remove = pd.DataFrame(expect_collection_expandTime['remove'])
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        testData = to_parquet(
            'test.parquet',
            tsc,
            expandCategory= False,
            expandTime=True,
            preprocessType= 'pad'
            )
        testData = pq.read_table('test.parquet').to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_pad,check_dtype=False)
        testData = to_parquet(
            'test.parquet',
            tsc,
            expandCategory= False,
            expandTime=True,
            preprocessType= 'remove'
            )
        testData = pq.read_table('test.parquet').to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_remove,check_dtype=False)
        with pytest.raises(ValueError):
            timeSeries = to_parquet('test.parquet',tsc,False,True,'ignore')
        os.remove('test.parquet')    


    def test_to_parquet_collection_expandCategory(self,dictList_collection,expect_collection_expandCategory):
        data = dictList_collection
        expandTime_pad = pd.DataFrame(expect_collection_expandCategory['pad'])
        expandTime_remove = pd.DataFrame(expect_collection_expandCategory['remove'])
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        testData = to_parquet(
            'test.parquet',
            tsc,
            expandCategory= True,
            expandTime=False,
            preprocessType= 'pad'
            )
        testData = pq.read_table('test.parquet').to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_pad,check_dtype=False)
        testData = to_parquet(
            'test.parquet',
            tsc,
            expandCategory= True,
            expandTime=False,
            preprocessType= 'remove'
            )
        testData = pq.read_table('test.parquet').to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_remove,check_dtype=False)
        with pytest.raises(ValueError):
            timeSeries = to_parquet('test.parquet',tsc,True,False,'ignore')    
        os.remove('test.parquet')

    def test_to_parquet_collection_expandFull(self,dictList_collection,expect_collection_expandFull):
        data = dictList_collection
        expandTime_pad = pd.DataFrame(expect_collection_expandFull['pad'])
        expandTime_remove = pd.DataFrame(expect_collection_expandFull['remove'])
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        testData = to_parquet(
            'test.parquet',
            tsc,
            expandCategory= True,
            expandTime=True,
            preprocessType= 'pad'
            )
        testData = pq.read_table('test.parquet').to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_pad,check_dtype=False)
        testData = to_parquet(
            'test.parquet',
            tsc,
            expandCategory= True,
            expandTime=True,
            preprocessType= 'remove'
            )
        testData = pq.read_table('test.parquet').to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_remove,check_dtype=False)
        with pytest.raises(ValueError):
            timeSeries = to_pandas('test.parquet',tsc,True,True,'ignore')

    def test_to_parquet_collection_noExpand(self,dictList_collection,expect_collection_noExpand):
        data = dictList_collection
        expandTime_ignore = pd.DataFrame(expect_collection_noExpand['ignore'])
        expandTime_pad = pd.DataFrame(expect_collection_noExpand['pad'])
        expandTime_remove = pd.DataFrame(expect_collection_noExpand['remove'])
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        testData = to_parquet(
            'test.parquet',
            tsc,
            expandCategory= False,
            expandTime=False,
            preprocessType= 'pad'
            )
        testData = pq.read_table('test.parquet').to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_pad,check_dtype=False)
        testData = to_parquet(
            'test.parquet',
            tsc,
            expandCategory= False,
            expandTime=False,
            preprocessType= 'remove'
            )
        testData = pq.read_table('test.parquet').to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_remove,check_dtype=False)
        testData = to_parquet(
            'test.parquet',
            tsc,
            expandCategory= False,
            expandTime=False,
            preprocessType= 'ignore'
            )
        testData = pq.read_table('test.parquet').to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_ignore,check_dtype=False)
        os.remove('test.parquet')

    def test_to_parquet_seperateLabels_single(self,dictList_single,expect_single_seperateLabel):
        data = dictList_single
        expectedX, expectedY = expect_single_seperateLabel
        expectedX = pd.DataFrame(expectedX)
        expectedY = pd.DataFrame(expectedY)
        tsd = Time_Series_Data(data,'time')
        tsd.set_labels([1,2],'data_label')
        to_parquet(['test.parquet','label.parquet'],tsd,False,False,'ignore',True)
        x = pq.read_table('test.parquet').to_pandas()
        y = pq.read_table('label.parquet').to_pandas()
        print(x)
        print(y)
        pd.testing.assert_frame_equal(x,expectedX,check_dtype=False)
        pd.testing.assert_frame_equal(y,expectedY,check_dtype=False)
        os.remove('test.parquet')
        os.remove('label.parquet')

    def test_to_parquet_seperateLabels_collection(self,dictList_collection,expect_collection_seperateLabel):
        data = dictList_collection
        expectedX, expectedY = expect_collection_seperateLabel
        expectedX = pd.DataFrame(expectedX)
        expectedY = pd.DataFrame(expectedY)
        tsd = Time_Series_Data(data,'time')
        tsd = tsd.set_labels([1,2,1,2],'data_label')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        to_parquet(['test.parquet','label.parquet'],tsc,False,False,'ignore',True)
        x = pq.read_table('test.parquet').to_pandas()
        y = pq.read_table('label.parquet').to_pandas()
        print(y)
        pd.testing.assert_frame_equal(x,expectedX,check_dtype=False)
        pd.testing.assert_frame_equal(y,expectedY,check_dtype=False) 
        os.remove('test.parquet')
        os.remove('label.parquet')

    def test_to_parquet_single_sequence(self,seq_single):
        data = seq_single
        df= pd.DataFrame(data)
        tsd = Time_Series_Data(data,'time')
        to_parquet('test.parquet',tsd,False,False,'ignore',False)
        test = pq.read_table('test.parquet').to_pandas()
        pd.testing.assert_frame_equal(test,df,False)
        os.remove('test.parquet')

    def test_to_parquet_collection_sequence(self,seq_collection,expect_seq_collection):
        data = seq_collection
        df = pd.DataFrame(data)
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        to_parquet('test.parquet',tsc,False,False,'ignore')
        test = pq.read_table('test.parquet').to_pandas()
        pd.testing.assert_frame_equal(df,test,False)
        to_parquet('test.parquet',tsc,True,True,'ignore')
        test = pq.read_table('test.parquet').to_pandas()
        full = pd.DataFrame(expect_seq_collection)
        print(test)
        print(full)
        test = test.reindex(sorted(df.columns), axis=1)
        full = full.reindex(sorted(df.columns), axis=1)
        pd.testing.assert_frame_equal(test,full,False)
        os.remove('test.parquet')


class Test_Generator_IO:
    def test_from_generator(self):
        pass

    def test_to_generator(self):
        pass


class Test_Feather_IO:
    def test_from_feather_single(self,dictList_single):
        data = dictList_single
        df = pd.DataFrame(dictList_single)
        table = pa.Table.from_pandas(df)
        pf.write_feather(table,'test.feather')
        testData = from_feather('test.feather','time',None)
        tsd = Time_Series_Data(data,'time')
        assert tsd == testData
        os.remove('test.feather')

    def test_from_feather_collection(self,dictList_collection):
        data = dictList_collection
        df = pd.DataFrame(dictList_collection)
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        table = pa.Table.from_pandas(df)
        pf.write_feather(table,'test_collection.feather')
        testData = from_feather('test_collection.feather','time','category')
        assert tsc == testData
        os.remove('test_collection.feather')

###########
    def test_to_feather_single(self,dictList_single,expect_single_expandTime):
        data = dictList_single
        df = pd.DataFrame(data)
        expandTime = pd.DataFrame(expect_single_expandTime)
        tsd = Time_Series_Data(data,'time')
        to_feather(
            'test.feather',
            tsd,
            expandCategory= None,
            expandTime=False,
            preprocessType= None
            )
        testData = pf.read_table('test.feather').to_pandas()
        pd.testing.assert_frame_equal(testData,df,check_dtype=False)
        to_feather(
            'test.feather',
            tsd,
            expandCategory= None,
            expandTime=True,
            preprocessType= None
            )
        testData = pf.read_table('test.feather').to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime,check_dtype=False)
        os.remove('test.feather')

    def test_to_feather_collection_expandTime(self,dictList_collection,expect_collection_expandTime):
        data = dictList_collection
        expandTime_pad = pd.DataFrame(expect_collection_expandTime['pad'])
        expandTime_remove = pd.DataFrame(expect_collection_expandTime['remove'])
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        testData = to_feather(
            'test.feather',
            tsc,
            expandCategory= False,
            expandTime=True,
            preprocessType= 'pad'
            )
        testData = pf.read_feather('test.feather')
        pd.testing.assert_frame_equal(testData,expandTime_pad,check_dtype=False)
        testData = to_feather(
            'test.feather',
            tsc,
            expandCategory= False,
            expandTime=True,
            preprocessType= 'remove'
            )
        testData = pf.read_feather('test.feather')
        pd.testing.assert_frame_equal(testData,expandTime_remove,check_dtype=False)
        with pytest.raises(ValueError):
            timeSeries = to_feather('test.feather',tsc,False,True,'ignore')
        os.remove('test.feather')    


    def test_to_feather_collection_expandCategory(self,dictList_collection,expect_collection_expandCategory):
        data = dictList_collection
        expandTime_pad = pd.DataFrame(expect_collection_expandCategory['pad'])
        expandTime_remove = pd.DataFrame(expect_collection_expandCategory['remove'])
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        testData = to_feather(
            'test.feather',
            tsc,
            expandCategory= True,
            expandTime=False,
            preprocessType= 'pad'
            )
        testData = pf.read_table('test.feather').to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_pad,check_dtype=False)
        testData = to_feather(
            'test.feather',
            tsc,
            expandCategory= True,
            expandTime=False,
            preprocessType= 'remove'
            )
        testData = pf.read_table('test.feather').to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_remove,check_dtype=False)
        with pytest.raises(ValueError):
            timeSeries = to_feather('test.feather',tsc,True,False,'ignore')    
        os.remove('test.feather')

    def test_to_feather_collection_expandFull(self,dictList_collection,expect_collection_expandFull):
        data = dictList_collection
        expandTime_pad = pd.DataFrame(expect_collection_expandFull['pad'])
        expandTime_remove = pd.DataFrame(expect_collection_expandFull['remove'])
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        testData = to_feather(
            'test.feather',
            tsc,
            expandCategory= True,
            expandTime=True,
            preprocessType= 'pad'
            )
        testData = pf.read_table('test.feather').to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_pad,check_dtype=False)
        testData = to_feather(
            'test.feather',
            tsc,
            expandCategory= True,
            expandTime=True,
            preprocessType= 'remove'
            )
        testData = pf.read_table('test.feather').to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_remove,check_dtype=False)
        with pytest.raises(ValueError):
            timeSeries = to_pandas('test.feather',tsc,True,True,'ignore')

    def test_to_feather_collection_noExpand(self,dictList_collection,expect_collection_noExpand):
        data = dictList_collection
        expandTime_ignore = pd.DataFrame(expect_collection_noExpand['ignore'])
        expandTime_pad = pd.DataFrame(expect_collection_noExpand['pad'])
        expandTime_remove = pd.DataFrame(expect_collection_noExpand['remove'])
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        testData = to_feather(
            'test.feather',
            tsc,
            expandCategory= False,
            expandTime=False,
            preprocessType= 'pad'
            )
        testData = pf.read_table('test.feather').to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_pad,check_dtype=False)
        testData = to_feather(
            'test.feather',
            tsc,
            expandCategory= False,
            expandTime=False,
            preprocessType= 'remove'
            )
        testData = pf.read_table('test.feather').to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_remove,check_dtype=False)
        testData = to_feather(
            'test.feather',
            tsc,
            expandCategory= False,
            expandTime=False,
            preprocessType= 'ignore'
            )
        testData = pf.read_table('test.feather').to_pandas()
        pd.testing.assert_frame_equal(testData,expandTime_ignore,check_dtype=False)
        os.remove('test.feather')

    def test_to_feather_seperateLabels_single(self,dictList_single,expect_single_seperateLabel):
        data = dictList_single
        expectedX, expectedY = expect_single_seperateLabel
        expectedX = pd.DataFrame(expectedX)
        expectedY = pd.DataFrame(expectedY)
        tsd = Time_Series_Data(data,'time')
        tsd.set_labels([1,2],'data_label')
        to_feather(['test.feather','label.feather'],tsd,False,False,'ignore',True)
        x = pf.read_table('test.feather').to_pandas()
        y = pf.read_table('label.feather').to_pandas()
        print(x)
        print(y)
        pd.testing.assert_frame_equal(x,expectedX,check_dtype=False)
        pd.testing.assert_frame_equal(y,expectedY,check_dtype=False)
        os.remove('test.feather')
        os.remove('label.feather')

    def test_to_feather_seperateLabels_collection(self,dictList_collection,expect_collection_seperateLabel):
        data = dictList_collection
        expectedX, expectedY = expect_collection_seperateLabel
        expectedX = pd.DataFrame(expectedX)
        expectedY = pd.DataFrame(expectedY)
        tsd = Time_Series_Data(data,'time')
        tsd = tsd.set_labels([1,2,1,2],'data_label')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        to_feather(['test.feather','label.feather'],tsc,False,False,'ignore',True)
        x = pf.read_table('test.feather').to_pandas()
        y = pf.read_table('label.feather').to_pandas()
        print(y)
        pd.testing.assert_frame_equal(x,expectedX,check_dtype=False)
        pd.testing.assert_frame_equal(y,expectedY,check_dtype=False) 
        os.remove('test.feather')
        os.remove('label.feather')

    def test_to_feather_single_sequence(self,seq_single):
        data = seq_single
        df= pd.DataFrame(data)
        tsd = Time_Series_Data(data,'time')
        to_feather('test.feather',tsd,False,False,'ignore',False,2)
        test = pf.read_table('test.feather').to_pandas()
        pd.testing.assert_frame_equal(test,df,False)
        os.remove('test.feather')

    def test_to_feather_collection_sequence(self,seq_collection,expect_seq_collection):
        data = seq_collection
        df = pd.DataFrame(data)
        tsd = Time_Series_Data(data,'time')
        tsc = Time_Series_Data_Collection(tsd,'time','category')
        to_feather('test.feather',tsc,False,False,'ignore',False,2)
        test = pf.read_table('test.feather').to_pandas()
        pd.testing.assert_frame_equal(df,test,False)
        to_feather('test.feather',tsc,True,True,'ignore',False,2)
        test = pf.read_table('test.feather').to_pandas()
        full = pd.DataFrame(expect_seq_collection)
        test = test.reindex(sorted(df.columns), axis=1)
        full = full.reindex(sorted(df.columns), axis=1)
        pd.testing.assert_frame_equal(test,full,False)
        os.remove('test.feather')