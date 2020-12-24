import numpy as np
import collections
import pandas as pd
from sklearn.base import (BaseEstimator, TransformerMixin)
from time_series_transform.io.numpy import (from_numpy,to_numpy)
from time_series_transform.io.pandas import (from_pandas,to_pandas)
from time_series_transform.io.parquet import (from_parquet, to_parquet)
from time_series_transform.stock_transform.stock_transfromer import Stock_Transformer
from time_series_transform.transform_core_api.time_series_transformer import Time_Series_Transformer


class Base_Time_Series_Transformer(BaseEstimator, TransformerMixin):

    def __init__(self,time_col,category_col=None,len_preprocessing = 'ignore',remove_time=True,remove_category=True,remove_org_data=True,cache_data_path = None):
        self._time_col = time_col 
        self._category_col = category_col
        self._time_series_cache = None
        self._len_preprocessing = len_preprocessing
        self._remove_time = remove_time
        self.cache_data_path = cache_data_path
        self.remove_org_data = remove_org_data
        self.remove_category = remove_category
        self.time_series_data = None
        self.category_cache= None

    def _cache_data(self,time_series_data):
        return to_parquet(self.cache_data_path,time_series_data,False,False,'ignore')

    def _to_time_series_data(self,X):
        if isinstance(X, pd.DataFrame):
            self._time_series_cache = X[self._time_col].tolist()
            time_series_data = from_pandas(X,self._time_col,self._category_col)
            if self._category_col is not None:
                self.category_cache = X[self._category_col].tolist()
        else:
            self._time_series_cache = list(X[:,self._time_col])
            time_series_data = from_numpy(X,self._time_col,self._category_col)
            if self._category_col is not None:
                self.category_cache = list(X[:,self._category_col])
        return time_series_data

    def _check_time_not_exist(self,timeList,categoryList):
        checkedList = []
        if categoryList is None:
            for i in timeList:
                if i not in self._time_series_cache:
                    checkedList.append(True)
                    continue
                checkedList.append(False)
        else:
            tmpDict = collections.defaultdict(list)
            for c,t in zip(self.category_cache,self._time_series_cache):
                tmpDict[c].append(t)
            for t,c in zip(timeList,categoryList):
                if t not in tmpDict[c]:
                    checkedList.append(True)
                    continue
                checkedList.append(False)
        return checkedList

    def fit(self,X,y = None):
        time_series_data = self._to_time_series_data(X)
        if self.cache_data_path is not None:
            self._cache_data(time_series_data)
            return self
        self.time_series_data = time_series_data
        return self 


    def transform(self,X,y = None):
        X_category = None
        if self.cache_data_path is not None:
            df = pd.read_parquet(self.cache_data_path)
        else:
            df = to_pandas(self.time_series_data,False,False,'ignore')
        X_time, X_category, X_header,new_df,check_list = self._prep_transform_data(X, X_category)   
        df = df.append(pd.DataFrame(new_df),ignore_index = True)
        tst = Time_Series_Transformer.from_pandas(
            df,
            self._time_col,
            self._category_col
            )
        if self._category_col is None:
            return tst,X_time,X_header,None
        return tst,X_time,X_header,X_category

    def _prep_transform_data(self, X, X_category):
        if isinstance(X,pd.DataFrame):
            X_time = X[self._time_col].tolist()
            if self._category_col is None:
                X_header = X.drop(self._time_col,axis =1).columns.tolist()
            else:
                X_header = X.drop(self._time_col,axis =1).drop(self._category_col,axis =1).columns.tolist()
                X_category = X[self._category_col].tolist()
            check_list = self._check_time_not_exist(X_time,X_category)
            new_df = X[check_list]
        else:
            X_time = list(X[:,self._time_col])
            if self._category_col is not None:
                X_category = list(X[:,self._category_col])
            X_header=[]
            for i in range(X.shape[1]):
                if i != int(self._time_col):
                    if self._category_col is not None:
                        if i == int(self._category_col):
                            continue
                    X_header.append(i)   
            check_list = self._check_time_not_exist(X_time,X_category)
            new_df = pd.DataFrame(X[check_list,:])
        return X_time, X_category, X_header,new_df, check_list

    def _transform_output_wrapper(self,df,X_category,X_time,X_header):
        if X_category is None:
            df = df[df[self._time_col].isin(X_time)]
        else:
            tmpdf = None
            tmpDict = collections.defaultdict(list)
            for ix,v in zip(X_time,X_category):
                tmpDict[v].append(ix)
            for i in tmpDict:
                if tmpdf is None:
                    tmpdf = df[df[self._category_col]==i][df[self._time_col].isin(tmpDict[i])]
                    continue
                tmpdf = tmpdf.append(df[df[self._category_col]==i][df[self._time_col].isin(tmpDict[i])])
            df = tmpdf
        if self.remove_category and self._category_col is not None:
            df = df.drop(self._category_col,axis =1)
        if self._remove_time:
            df = df.drop(self._time_col,axis =1)
        if self.remove_org_data:
            df= df.drop(X_header,axis =1)
        return df.values        

    def get_time_series_index_cache (self):
        return self._time_series_cache


class Lag_Transformer(Base_Time_Series_Transformer):
    def __init__(
                self,
                lag_nums,
                time_col,
                category_col=None,
                remove_time = True,
                remove_category=True,
                remove_org_data=True,
                cache_data_path=None):
        super().__init__(time_col,category_col,'ignore',remove_time,remove_category,remove_org_data,cache_data_path)
        if not isinstance(lag_nums,list):
            self.lag_nums = [lag_nums]
        else:
            self.lag_nums = lag_nums

    def fit(self,X,y=None):
        super().fit(X)
        return self

    def transform(self,X,y=None):
        tst,X_time,X_header,X_category = super().transform(X,y)
        for i in self.lag_nums:
            tst = tst.make_lag(X_header,lagNum=i,suffix=None)
        df = tst.to_pandas()
        return self._transform_output_wrapper(df,X_category,X_time,X_header)


class Function_Transformer(Base_Time_Series_Transformer):
    def __init__(
                self,
                func,
                inputLabels,
                time_col,
                category_col=None,
                remove_time = True,
                remove_category=True,
                remove_org_data=True,
                cache_data_path=None,
                parameterDict={}):
        super().__init__(time_col,category_col,'ignore',remove_time,remove_category,remove_org_data,cache_data_path)
        self.parameterDict = parameterDict
        self.parameterDict['func']= func
        self.parameterDict['inputLabels']= inputLabels
        self.parameterDict['newName']='newName'
        
    def fit(self,X,y=None):
        super().fit(X)
        return self

    def transform(self,X,y=None):
        tst,X_time,X_header,X_category = super().transform(X,y)
        tst = tst.transform(**self.parameterDict)
        df = tst.to_pandas()
        return self._transform_output_wrapper(df,X_category,X_time,X_header)


class Base_Stock_Time_Series_Transform(Base_Time_Series_Transformer):
    def __init__(
                self,
                time_col,
                category_col=None,
                len_preprocessing = 'ignore',
                remove_time=True,
                remove_category=True,
                remove_org_data=True,
                cache_data_path = None,
                High='High',
                Low='Low',
                Close='Close',
                Open='Open',
                Volume='Volume'):
        super().__init__(time_col,category_col,len_preprocessing,remove_time,remove_category,remove_org_data,cache_data_path)
        self.high = High
        self.low = Low
        self.open = Open
        self.close = Close
        self.volume = Volume


    def transform(self,X,y= None):
        tst,X_time,X_header,X_category = super().transform(X,y)
        tst = Stock_Transformer.from_time_series_transformer(
            tst,
            High = self.high,
            Low = self.low,
            Close = self.close,
            Open = self.open,
            Volume = self.volume
            )        
        return tst, X_time,X_header,X_category



class Stock_Technical_Indicator_Transformer(Base_Stock_Time_Series_Transform):
    def __init__(
                self,
                strategy,
                time_col,
                symbol_col=None,
                remove_time = True,
                remove_category=True,
                remove_org_data=True,
                cache_data_path=None,
                High='High',
                Low='Low',
                Close='Close',
                Open='Open',
                Volume='Volume',
                n_jobs = 1,
                verbose = 0,
                backend='loky'):

        super().__init__(time_col,symbol_col,'ignore',remove_time,remove_category,remove_org_data,cache_data_path,High,Low,Close,Open,Volume)
        self.strategy = strategy
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.backend = backend

    def fit(self,X,y=None):
        super().fit(X,y)
        return self

    def transform(self,X,y=None):
        tst, X_time,X_header,X_category = super().transform(X,y)
        tst = tst.get_technial_indicator(self.strategy,self.n_jobs,self.verbose,self.backend)
        df = tst.to_pandas()
        return self._transform_output_wrapper(df,X_category,X_time,X_header)


