import copy
import pytest
import numpy as np
import pandas as pd
from time_series_transform.transform_core_api import *
from time_series_transform.transform_core_api.util import *
from time_series_transform.transform_core_api.base import *
from time_series_transform.transform_core_api.time_series_transformer import *
import pyarrow as pa
import os


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
def expect_single_lag():
    return {
        'time':[1,2],
        'data':[1,2],
        'data_lag_1':[np.nan,1]
    }

@pytest.fixture(scope = 'class')
def expect_single_lead():
    return {
        'time':[1,2],
        'data':[1,2],
        'data_lead_1':[2,np.nan]
    }

@pytest.fixture(scope = 'class')
def expect_single_lag_sequence():
    return {
        'time':[1,2],
        'data':[1,2],
        'data_lag_1':[[np.nan],[1]]
    }

@pytest.fixture(scope = 'class')
def expect_single_lead_sequence():
    return {
        'time':[1,2],
        'data':[1,2],
        'data_lead_1':[[2],[np.nan]]
    }

@pytest.fixture(scope = 'class')
def expect_single_identical_sequence():
    return {
        'time':[1,2],
        'data':[1,2],
        'data_identical_2':[[1,1],[2,2]]
    }

@pytest.fixture(scope = 'class')
def expect_collection_lag():
    return {
        'time': [1,2,1,3],
        'data':[1,2,1,2],
        'data_lag_1':[np.nan,1,np.nan,1],
        'category':[1,1,2,2],
    }

@pytest.fixture(scope = 'class')
def expect_collection_lead():
    return {
        'time': [1,2,1,3],
        'data':[1,2,1,2],
        'data_lead_1':[2,np.nan,2,np.nan],
        'category':[1,1,2,2],
    }

@pytest.fixture(scope = 'class')
def expect_collection_lag_sequence():
    return {
        'time': [1,2,1,3],
        'data':[1,2,1,2],
        'data_lag_1':[[np.nan],[1],[np.nan],[1]],
        'category':[1,1,2,2],
    }

@pytest.fixture(scope = 'class')
def expect_collection_lead_sequence():
    return {
        'time': [1,2,1,3],
        'data':[1,2,1,2],
        'data_lead_1':[[2],[np.nan],[2],[np.nan]],
        'category':[1,1,2,2],
    }

@pytest.fixture(scope = 'class')
def expect_collection_identity_sequence():
    return {
        'time': [1,2,1,3],
        'data':[1,2,1,2],
        'data_identical_2':[[1,1],[2,2],[1,1],[2,2]],
        'category':[1,1,2,2],
    }

@pytest.fixture(scope = 'class')
def expect_single_stack_sequence():
    return {
        'time':[1,2],
        'data':[1,2],
        'data_lag_1':[[np.nan],[1]],
        'stack_data':[[[np.nan,np.nan]],[[1,1]]]
    }

@pytest.fixture(scope = 'class')
def expect_collection_stack_sequence():
    return {
        'time': [1,2,1,3],
        'data':[1,2,1,2],
        'data_lag_1':[[np.nan],[1],[np.nan],[1]],
        'stack_data':[[[np.nan,np.nan]],[[1,1]],[[np.nan,np.nan]],[[1,1]]],
        'category':[1,1,2,2]        
    }


@pytest.fixture(scope = 'class')
def single_na_test():
    return {
        'test':{
            'time':[1,2,3,4],
            'data':[np.nan,1,1,1],
            'data_seq':[[1,2],[np.nan,1],[1,2],[1,2]]
        },
        'res':{
            'time':[3,4],
            'data':[1,1],
            'data_seq':[[1,2],[1,2]]
        }
    }

@pytest.fixture(scope = 'class')
def collection_na_test():
    return {
        'test':{
            'time':[1,2,3,1,2,3],
            'data':[1,2,3,1,np.nan,3],
            'data_seq':[[1,2],[np.nan,1],[1,2],[1,2],[np.nan,1],[1,2]],
            'category':[1,1,1,2,2,2]
        },
        'res':{
            'time':[1,3,1,3],
            'data':[1,3,1,3],
            'data_seq':[[1,2],[1,2],[1,2],[1,2]],
            'category':[1,1,2,2]
        }
    }

class Test_time_series_transform:
    def test_from_pandas(self,dictList_single,dictList_collection):
        data = dictList_single
        result = Time_Series_Transformer(data,'time',None)
        assert Time_Series_Transformer.from_pandas(pd.DataFrame(data),'time',None) == result
        data = dictList_collection
        result = Time_Series_Transformer(data,'time','category')
        assert Time_Series_Transformer.from_pandas(pd.DataFrame(data),'time','category') == result
                
    def test_from_numpy(self,dictList_single,dictList_collection):
        data = pd.DataFrame(pd.DataFrame(dictList_single).values)
        result = Time_Series_Transformer(data,0,None)
        assert Time_Series_Transformer.from_numpy(pd.DataFrame(dictList_single).values,0,None) == result
        data = pd.DataFrame(pd.DataFrame(dictList_collection).values)
        result = Time_Series_Transformer(data,0,None)
        assert Time_Series_Transformer.from_numpy(pd.DataFrame(dictList_collection).values,0,None) == result


    def test_to_pandas(self,dictList_single,dictList_collection):
        data = dictList_single
        result = pd.DataFrame(data)
        df = Time_Series_Transformer(data,'time',None).to_pandas()
        pd.testing.assert_frame_equal(df,result,False)
        data = dictList_collection
        result = pd.DataFrame(data)
        df = Time_Series_Transformer(data,'time','category').to_pandas()
        pd.testing.assert_frame_equal(df,result,False)


    def test_to_numpy(self,dictList_single,dictList_collection):
        data = dictList_single
        result = pd.DataFrame(data).values
        numpyData = Time_Series_Transformer(data,'time',None).to_numpy()
        np.testing.assert_equal(numpyData,result)
        data = dictList_collection
        result = pd.DataFrame(data).values
        numpyData = Time_Series_Transformer(data,'time',None).to_numpy()
        np.testing.assert_equal(numpyData,result)


    def test_to_arrow_table(self,dictList_single,dictList_collection):
        df = pd.DataFrame(dictList_single)
        tst = Time_Series_Transformer.from_pandas(df,'time',None)
        test = tst.to_arrow_table().to_pandas()
        pd.testing.assert_frame_equal(test,df)
        df = pd.DataFrame(dictList_collection)
        tst = Time_Series_Transformer.from_pandas(df,'time','category')
        test = tst.to_arrow_table().to_pandas()
        pd.testing.assert_frame_equal(test,df)

    def test_from_arrow_table(self,dictList_single,dictList_collection):
        df = pd.DataFrame(dictList_single)
        arrowTable = pa.Table.from_pandas(df)
        tst = Time_Series_Transformer.from_arrow_table(arrowTable,'time',None)
        test = Time_Series_Transformer.from_pandas(df,'time',None)
        assert tst == test
        df = pd.DataFrame(dictList_collection)
        arrowTable = pa.Table.from_pandas(df)
        tst = Time_Series_Transformer.from_arrow_table(arrowTable,'time','category')
        test = Time_Series_Transformer.from_pandas(df,'time','category')
        assert tst == test

    def test_to_feather(self,dictList_single,dictList_collection):
        df = pd.DataFrame(dictList_single)
        tst = Time_Series_Transformer.from_pandas(df,'time',None)
        tst.to_feather('./data.feather')
        test = pd.read_feather('./data.feather')
        pd.testing.assert_frame_equal(test,df)
        df = pd.DataFrame(dictList_collection)
        tst = Time_Series_Transformer.from_pandas(df,'time','category')
        tst.to_feather('./data.feather')
        test = pd.read_feather('./data.feather')
        pd.testing.assert_frame_equal(test,df)
        os.remove('./data.feather')

    def test_from_feather(self,dictList_single,dictList_collection):
        df = pd.DataFrame(dictList_single)
        df.to_feather('./data.feather')
        test = Time_Series_Transformer.from_feather('./data.feather','time',None)
        tst = Time_Series_Transformer.from_pandas(df,'time',None)
        assert test == tst
        df = pd.DataFrame(dictList_collection)
        df.to_feather('./data.feather')
        test = Time_Series_Transformer.from_feather('./data.feather','time','category')
        tst = Time_Series_Transformer.from_pandas(df,'time','category')
        assert test == tst
        os.remove('./data.feather')

    def test_to_parquet(self,dictList_single,dictList_collection):
        df = pd.DataFrame(dictList_single)
        tst = Time_Series_Transformer.from_pandas(df,'time',None)
        tst.to_parquet('./data.parquet')
        test = pd.read_parquet('./data.parquet')
        pd.testing.assert_frame_equal(test,df)
        os.remove('./data.parquet')
        df = pd.DataFrame(dictList_collection)
        tst = Time_Series_Transformer.from_pandas(df,'time','category')
        tst.to_parquet('./data.parquet')
        test = pd.read_parquet('./data.parquet')
        pd.testing.assert_frame_equal(test,df)
        os.remove('./data.parquet')

    def test_from_parquet(self,dictList_single,dictList_collection):
        df = pd.DataFrame(dictList_single)
        df.to_parquet('./data.parquet')
        test = Time_Series_Transformer.from_parquet('./data.parquet','time',None)
        tst = Time_Series_Transformer.from_pandas(df,'time',None)
        assert test == tst
        os.remove('./data.parquet')
        df = pd.DataFrame(dictList_collection)
        df.to_parquet('./data.parquet')
        test = Time_Series_Transformer.from_parquet('./data.parquet','time','category')
        tst = Time_Series_Transformer.from_pandas(df,'time','category')
        assert test == tst
        os.remove('./data.parquet')


    def test_to_dict(self,dictList_single,dictList_collection):
        data = dictList_single
        dictData = Time_Series_Transformer(data,'time',None).to_dict()
        assert len(dictData) == len(data)
        for i in data:
            assert data[i] == dictData[i].tolist()
        data = dictList_collection
        dictData = Time_Series_Transformer(data,'time',None).to_dict()
        assert len(dictData) == len(data)
        for i in data:
            assert data[i] == dictData[i].tolist()

    def test_single_make_lag(self,dictList_single,expect_single_lag):
        data = dictList_single
        expectDf = pd.DataFrame(expect_single_lag)
        tst = Time_Series_Transformer(data,'time',None)
        tst.make_lag('data',1,'_lag_')
        df = tst.to_pandas()
        pd.testing.assert_frame_equal(df,expectDf,False)

    def test_single_make_lead(self,dictList_single,expect_single_lead):
        data = dictList_single
        expectDf = pd.DataFrame(expect_single_lead)
        tst = Time_Series_Transformer(data,'time',None)
        tst.make_lead('data',1,'_lead_')
        df = tst.to_pandas()
        pd.testing.assert_frame_equal(df,expectDf,False)

    def test_single_lag_sequence(self,dictList_single,expect_single_lag_sequence):
        data = dictList_single
        expectDf = pd.DataFrame(expect_single_lag_sequence)
        tst = Time_Series_Transformer(data,'time',None)
        tst.make_lag_sequence('data',1,1,'_lag_')
        df = tst.to_pandas()
        pd.testing.assert_frame_equal(df,expectDf,False)


    def test_single_lead_sequence(self,dictList_single,expect_single_lead_sequence):
        data = dictList_single
        expectDf = pd.DataFrame(expect_single_lead_sequence)
        tst = Time_Series_Transformer(data,'time',None)
        tst.make_lead_sequence('data',1,1,'_lead_')
        df = tst.to_pandas()
        pd.testing.assert_frame_equal(df,expectDf,False)

    def test_single_identical_sequence(self,dictList_single,expect_single_identical_sequence):
        data = dictList_single
        expectDf = pd.DataFrame(expect_single_identical_sequence)
        tst = Time_Series_Transformer(data,'time',None)
        tst.make_identical_sequence('data',2,'_identical_')
        df = tst.to_pandas()
        pd.testing.assert_frame_equal(df,expectDf,False)

    def test_collection_lag(self,dictList_collection,expect_collection_lag):
        data = dictList_collection
        expectResults = expect_collection_lag
        expectResults = pd.DataFrame(expectResults)
        tst = Time_Series_Transformer(data,'time','category')
        tst.make_lag('data',1,'_lag_')
        df = tst.to_pandas()
        pd.testing.assert_frame_equal(df,expectResults,False)
        

    def test_collection_lead(self,dictList_collection,expect_collection_lead):
        data = dictList_collection
        expectResults = expect_collection_lead
        expectResults = pd.DataFrame(expectResults)
        tst = Time_Series_Transformer(data,'time','category')
        tst.make_lead('data',1,'_lead_')
        df = tst.to_pandas()
        pd.testing.assert_frame_equal(df,expectResults,False)

    def test_collection_lag_sequence(self,dictList_collection,expect_collection_lag_sequence):
        data = dictList_collection
        expectResults = expect_collection_lag_sequence
        expectResults = pd.DataFrame(expectResults)
        tst = Time_Series_Transformer(data,'time','category')
        tst.make_lag_sequence('data',1,1,'_lag_')
        df = tst.to_pandas()
        pd.testing.assert_frame_equal(df,expectResults,False)

    def test_collection_lead_sequence(self,dictList_collection,expect_collection_lead_sequence):
        data = dictList_collection
        expectResults = expect_collection_lead_sequence
        expectResults = pd.DataFrame(expectResults)
        tst = Time_Series_Transformer(data,'time','category')
        tst.make_lead_sequence('data',1,1,'_lead_')
        df = tst.to_pandas()
        pd.testing.assert_frame_equal(df,expectResults,False)

    def test_collection_identity_sequence(self,dictList_collection,expect_collection_identity_sequence):
        data = dictList_collection
        expectResults = expect_collection_identity_sequence
        expectResults = pd.DataFrame(expectResults)
        tst = Time_Series_Transformer(data,'time','category')
        tst.make_identical_sequence('data',2,'_identical_')
        df = tst.to_pandas()
        pd.testing.assert_frame_equal(df,expectResults,False)


    def test_collection_stack_sequence(self,dictList_collection,expect_collection_stack_sequence):
        data = dictList_collection
        expectResults = pd.DataFrame(expect_collection_stack_sequence)
        tst = Time_Series_Transformer(data,'time','category')
        tst = tst.make_lag_sequence('data',1,1,'_lag_')
        tst = tst.make_stack_sequence(['data_lag_1','data_lag_1'],'stack_data',-1)
        df = tst.to_pandas()
        pd.testing.assert_frame_equal(df,expectResults,False)


    def test_single_stack_sequence(self,dictList_single,expect_single_stack_sequence):
        data = dictList_single
        expectResults = pd.DataFrame(expect_single_stack_sequence)
        tst = Time_Series_Transformer(data,'time',None)
        tst = tst.make_lag_sequence('data',1,1,'_lag_')
        tst = tst.make_stack_sequence(['data_lag_1','data_lag_1'],'stack_data',-1)
        df = tst.to_pandas()
        pd.testing.assert_frame_equal(df,expectResults,False)

    def test_single_make_label(self,dictList_single):
        data = dictList_single
        tst = Time_Series_Transformer(data,'time',None)
        tst = tst.make_lag('data',1,'_lag_')
        tst = tst.make_label('data')
        print(tst.time_series_data)
        print(tst.time_series_data.labels)
        _,df= tst.to_pandas(sepLabel=True)
        assert df.data.tolist() == data['data']


    def test_collection_make_label(self,dictList_collection):
        data = dictList_collection
        tst = Time_Series_Transformer(data,'time','category')
        tst = tst.make_lag('data',1,'_lag_')
        tst = tst.make_label('data')
        _,df= tst.to_pandas(sepLabel=True)
        assert df.data.tolist() == data['data']


    def test_single_dropna(self,single_na_test):
        data = single_na_test['test']
        res = pd.DataFrame(single_na_test['res'])
        tst = Time_Series_Transformer(data,'time',None)
        tst = tst.dropna()
        pd.testing.assert_frame_equal(tst.to_pandas(),res,False)

    def test_collection_dropna(self,collection_na_test):
        data = collection_na_test['test']
        res = pd.DataFrame(collection_na_test['res'])
        tst = Time_Series_Transformer(data,'time','category')
        tst = tst.dropna()
        print(tst.to_pandas())
        pd.testing.assert_frame_equal(tst.to_pandas(),res,False)


