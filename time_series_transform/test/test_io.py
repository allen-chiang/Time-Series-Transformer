import copy
import pytest
import numpy as np
import pandas as pd
from time_series_transform.io.base import io_base
from time_series_transform.io.numpy import (from_numpy, to_numpy)
from time_series_transform.io.pandas import (from_pandas, to_pandas)
from time_series_transform.transform_core_api.base import (
    Time_Series_Data, Time_Series_Data_Collection)


@pytest.fixture('class')
def dictList_single():
    return {
        'time': [1, 2],
        'data': [1, 2]
    }

@pytest.fixture('class')
def dictList_collection():
    return {
        'time': [1,2,1,3],
        'data':[1,2,1,2],
        'category':[1,1,2,2]
    }

@pytest.fixture('class')
def expect_single_expandTime():
    return {
        'data_1':[1],
        'data_2':[2]
    }

@pytest.fixture('class')
def expect_single_seperateLabel():
    return [{
        'time': [1, 2],
        'data': [1, 2]
    },
    {
        'data_label': [1, 2]
    }]

@pytest.fixture('class')
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

@pytest.fixture('class')
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
@pytest.fixture('class')
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

@pytest.fixture('class')
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

@pytest.fixture('class')
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
        with pytest.raises(ValueError):
            timeSeries = io.from_collection(True,True,'ignore')
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
        with pytest.raises(ValueError):
            timeSeries = to_pandas(tsc,True,True,'ignore')

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
        with pytest.raises(ValueError):
            timeSeries = to_numpy(tsc,True,True,'ignore')

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

class Test_Arrow_IO:

    def test_to_arrow_table_single(self):
        pass

    def test_to_arrow_table_collection(self):
        pass

    



class Test_Generator_IO:
    def test_from_generator(self):
        pass

    def test_to_generator(self):
        pass