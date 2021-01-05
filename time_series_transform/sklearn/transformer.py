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
        """
        __init__ Base class for sklearn transformer implmemnting Time_Series_Transformer
        
        This class prepared the data into Time_Series_Transformer. It can also use parquet
        to save data for future transformation. Since transforming time series data is usually
        associated with past data, this class cache the past data and check whether it will be used
        during transformation
        
        Parameters
        ----------
        time_col : str or int
            the index of time series period column
        category_col : str or int, optional
            category column index, by default None
        len_preprocessing : ['ignore','pad','remove'] , optional
            data preprocessing for categories, by default 'ignore'
        remove_time : bool, optional
            whether to remove time column for output, by default True
        remove_category : bool, optional
            whether to remove category column for output, by default True
        remove_org_data : bool, optional
            whether to remove orign data for output, by default True
        cache_data_path : str, optional
            the path to cache data, by default None
        """
        self.time_col = time_col 
        self.category_col = category_col
        self.time_series_cache = None
        self.len_preprocessing = len_preprocessing
        self.remove_time = remove_time
        self.cache_data_path = cache_data_path
        self.remove_org_data = remove_org_data
        self.remove_category = remove_category
        self.time_series_data = None
        self.category_cache= None

    def _cache_data(self,time_series_data):
        return to_parquet(self.cache_data_path,time_series_data,False,False,'ignore')

    def _to_time_series_data(self,X):
        if isinstance(X, pd.DataFrame):
            self.time_series_cache = X[self.time_col].tolist()
            time_series_data = from_pandas(X,self.time_col,self.category_col)
            if self.category_col is not None:
                self.category_cache = X[self.category_col].tolist()
        else:
            self.time_series_cache = list(X[:,self.time_col])
            time_series_data = from_numpy(X,self.time_col,self.category_col)
            if self.category_col is not None:
                self.category_cache = list(X[:,self.category_col])
        return time_series_data

    def _check_time_not_exist(self,timeList,categoryList):
        checkedList = []
        if categoryList is None:
            for i in timeList:
                if i not in self.time_series_cache:
                    checkedList.append(True)
                    continue
                checkedList.append(False)
        else:
            tmpDict = collections.defaultdict(list)
            for c,t in zip(self.category_cache,self.time_series_cache):
                tmpDict[c].append(t)
            for t,c in zip(timeList,categoryList):
                if t not in tmpDict[c]:
                    checkedList.append(True)
                    continue
                checkedList.append(False)
        return checkedList

    def fit(self,X,y = None):
        """
        fit train model
        
        Parameters
        ----------
        X : pandas DataFrame or numpy ndArray
            input values
        y : depreciated not used, optional
            following sklearn convention (not used), by default None
        
        Returns
        -------
        self
        """
        time_series_data = self._to_time_series_data(X)
        if self.cache_data_path is not None:
            self._cache_data(time_series_data)
            return self
        self.time_series_data = time_series_data
        return self 


    def transform(self,X,y = None):
        """
        transform prepare the data as Time_Series_Transformer and other helper data
        
        Parameters
        ----------
        X : pandas DataFrame or numpy ndArray
            input values
        y : depreciated not used, optional
            following sklearn convention (not used), by default None
        
        Returns
        -------
        tst Time_Series_Transformer
            the output Time_Series_Transformer
        X_time list
            time column list
        X_header list
            column name list
        X_category list
            category name list

        """
        X_category = None
        if self.cache_data_path is not None:
            df = pd.read_parquet(self.cache_data_path)
        else:
            df = to_pandas(self.time_series_data,False,False,'ignore')
        X_time, X_category, X_header,new_df,check_list = self._prep_transform_data(X, X_category)   
        df = df.append(pd.DataFrame(new_df),ignore_index = True)
        tst = Time_Series_Transformer.from_pandas(
            df,
            self.time_col,
            self.category_col
            )
        if self.category_col is None:
            return tst,X_time,X_header,None
        return tst,X_time,X_header,X_category

    def _prep_transform_data(self, X, X_category):
        if isinstance(X,pd.DataFrame):
            X_time = X[self.time_col].tolist()
            if self.category_col is None:
                X_header = X.drop(self.time_col,axis =1).columns.tolist()
            else:
                X_header = X.drop(self.time_col,axis =1).drop(self.category_col,axis =1).columns.tolist()
                X_category = X[self.category_col].tolist()
            check_list = self._check_time_not_exist(X_time,X_category)
            new_df = X[check_list]
        else:
            X_time = list(X[:,self.time_col])
            if self.category_col is not None:
                X_category = list(X[:,self.category_col])
            X_header=[]
            for i in range(X.shape[1]):
                if i != int(self.time_col):
                    if self.category_col is not None:
                        if i == int(self.category_col):
                            continue
                    X_header.append(i)   
            check_list = self._check_time_not_exist(X_time,X_category)
            new_df = pd.DataFrame(X[check_list,:])
        return X_time, X_category, X_header,new_df, check_list

    def _transform_output_wrapper(self,df,X_category,X_time,X_header):
        """
        _transform_output_wrapper the helper function for transformed data output
        
        Parameters
        ----------
        df : pandas dataFrame
            transformerd data
        X_time list
            time column list
        X_header list
            column name list
        X_category list
            category name list
        
        Returns
        -------
        [type]
            [description]
        """
        if X_category is None:
            df = df[df[self.time_col].isin(X_time)]
        else:
            tmpdf = None
            tmpDict = collections.defaultdict(list)
            for ix,v in zip(X_time,X_category):
                tmpDict[v].append(ix)
            for i in tmpDict:
                if tmpdf is None:
                    tmpdf = df[df[self.category_col]==i][df[self.time_col].isin(tmpDict[i])]
                    continue
                tmpdf = tmpdf.append(df[df[self.category_col]==i][df[self.time_col].isin(tmpDict[i])])
            df = tmpdf
        if self.remove_category and self.category_col is not None:
            df = df.drop(self.category_col,axis =1)
        if self.remove_time:
            df = df.drop(self.time_col,axis =1)
        if self.remove_org_data:
            df= df.drop(X_header,axis =1)
        return df.values        

    def get_time_series_index_cache (self):
        """
        get_time_series_index_cache the fitted time series index
        help to see when is the latest timestamp of the model
        
        Returns
        -------
        list
            cached time series index
        """
        return self.time_series_cache


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
        """
        __init__ Transform input data into series of lag data
        
        Parameters
        ----------
        lag_nums : int or list of int
            lag period numbers
        time_col : str or int
            the index of time series period column
        category_col : str or int, optional
            category column index, by default None
        len_preprocessing : ['ignore','pad','remove'] , optional
            data preprocessing for categories, by default 'ignore'
        remove_time : bool, optional
            whether to remove time column for output, by default True
        remove_category : bool, optional
            whether to remove category column for output, by default True
        remove_org_data : bool, optional
            whether to remove orign data for output, by default True
        cache_data_path : str, optional
            the path to cache data, by default None
        """
        super().__init__(time_col,category_col,'ignore',remove_time,remove_category,remove_org_data,cache_data_path)
        if not isinstance(lag_nums,list):
            self.lag_nums = [lag_nums]
        else:
            self.lag_nums = lag_nums

    def fit(self,X,y=None):
        """
        fit train model
        
        Parameters
        ----------
        X : pandas DataFrame or numpy ndArray
            input values
        y : depreciated not used, optional
            following sklearn convention (not used), by default None
        
        Returns
        -------
        self
        """
        super().fit(X)
        return self

    def transform(self,X,y=None):
        """
        transform transforming lag data
        
        Parameters
        ----------
        X : pandas DataFrame or numpy ndArray
            input values
        y : depreciated not used, optional
            following sklearn convention (not used), by default None
        
        Returns
        -------
        numpy ndArray
            transformed data
        """
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
        """
        __init__ Function Transformer provides an implmentation for custom functions
        
        Parameters
        ----------
        func : function
            the data manipulation function
        inputLabels : str, numeric data or list of data or numeric data
            the input data columns passing to function
        time_col : str or int
            the index of time series period column
        category_col : str or int, optional
            category column index, by default None
        len_preprocessing : ['ignore','pad','remove'] , optional
            data preprocessing for categories, by default 'ignore'
        remove_time : bool, optional
            whether to remove time column for output, by default True
        remove_category : bool, optional
            whether to remove category column for output, by default True
        remove_org_data : bool, optional
            whether to remove orign data for output, by default True
        cache_data_path : str, optional
            the path to cache data, by default None
        parameterDict : dict, optional
            input parameters, by default {}
        """
        super().__init__(time_col,category_col,'ignore',remove_time,remove_category,remove_org_data,cache_data_path)
        self.parameterDict = parameterDict
        self.parameterDict['func']= func
        self.parameterDict['inputLabels']= inputLabels
        self.parameterDict['newName']='newName'
        
    def fit(self,X,y=None):
        """
        fit train model
        
        Parameters
        ----------
        X : pandas DataFrame or numpy ndArray
            input values
        y : depreciated not used, optional
            following sklearn convention (not used), by default None
        
        Returns
        -------
        self
        """
        super().fit(X)
        return self

    def transform(self,X,y=None):
        """
        transform transforming data
        
        Parameters
        ----------
        X : pandas DataFrame or numpy ndArray
            input values
        y : depreciated not used, optional
            following sklearn convention (not used), by default None
        
        Returns
        -------
        numpy ndArray
            transformed data
        """
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
        """
        __init__ The base class implmenting Stock_Transformer
        
        Parameters
        ----------
        time_col : str or int
            the index of time series period column
        category_col : str or int, optional
            category column index, by default None
        len_preprocessing : ['ignore','pad','remove'] , optional
            data preprocessing for categories, by default 'ignore'
        remove_time : bool, optional
            whether to remove time column for output, by default True
        remove_category : bool, optional
            whether to remove category column for output, by default True
        remove_org_data : bool, optional
            whether to remove orign data for output, by default True
        cache_data_path : str, optional
            the path to cache data, by default None
        High : str or int, optional
            the index or name for High, by default 'High'
        Low : str or int, optional
            the index or name for Low, by default 'Low'
        Close : str or int, optional
            the index or name for Close, by default 'Close'
        Open : str or int, optional
            the index or name for Open, by default 'Open'
        Volume : str or int, optional
            the index or name for Volume, by default 'Volume'
        """
        super().__init__(time_col,category_col,len_preprocessing,remove_time,remove_category,remove_org_data,cache_data_path)
        self.high = High
        self.low = Low
        self.open = Open
        self.close = Close
        self.volume = Volume


    def transform(self,X,y= None):
        """
        transform prepare data as Stock_Transformer and helper data
        
        Parameters
        ----------
        X : pandas DataFrame or numpy ndArray
            input values
        y : depreciated not used, optional
            following sklearn convention (not used), by default None
        
        Returns
        -------
        tst Stock_Transformer
            the output Stock_Transformer
        X_time list
            time column list
        X_header list
            column name list
        X_category list
            category name list
        """
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
        """
        __init__ transforming data into techinical indicator through pandas-ta api
        
        Note: when using this transformer in pipeline, it is important to understand whether
        the input data is numpy or pandas dataFrame. Open, Close, High, Low, and Volume column index
        must match with input sources.
        
        Parameters
        ----------
        strategy : Strategy
            pandas-ta strategy
        time_col : str
            the name of time_col
        symbolIx : str or int
            the symbol column index of the data
        remove_time : bool, optional
            [description], by default True
        remove_category : bool, optional
            [description], by default True
        remove_org_data : bool, optional
            [description], by default True
        cache_data_path : [type], optional
            [description], by default None
        High : str or int, optional
            the index or name for High, by default 'High'
        Low : str or int, optional
            the index or name for Low, by default 'Low'
        Close : str or int, optional
            the index or name for Close, by default 'Close'
        Open : str or int, optional
            the index or name for Open, by default 'Open'
        Volume : str or int, optional
            the index or name for Volume, by default 'Volume'
        n_jobs : int, optional
            number of processes (joblib), by default 1
        verbose : int, optional
            log level (joblib), by default 0
        backend : str, optional
            backend type (joblib), by default 'loky'
        """

        super().__init__(time_col,symbol_col,'ignore',remove_time,remove_category,remove_org_data,cache_data_path,High,Low,Close,Open,Volume)
        self.strategy = strategy
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.backend = backend

    def fit(self,X,y=None):
        """
        fit train model
        
        Parameters
        ----------
        X : pandas DataFrame or numpy ndArray
            input values
        y : depreciated not used, optional
            following sklearn convention (not used), by default None
        
        Returns
        -------
        self
        """
        super().fit(X,y)
        return self

    def transform(self,X,y=None):
        """
        transform transforming data according to the strategy
        
        Parameters
        ----------
        X : pandas DataFrame or numpy ndArray
            input values
        y : depreciated not used, optional
            following sklearn convention (not used), by default None
        
        Returns
        -------
        numpy ndArray
            transformed data
        """
        tst, X_time,X_header,X_category = super().transform(X,y)
        tst = tst.get_technial_indicator(self.strategy,self.n_jobs,self.verbose,self.backend)
        df = tst.to_pandas()
        return self._transform_output_wrapper(df,X_category,X_time,X_header)


