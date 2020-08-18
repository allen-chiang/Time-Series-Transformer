import gc
import uuid
import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq
from collections import defaultdict
from time_series_transform.transform_core_api.base import *



class Pandas_Time_Series_Tensor_Dataset(object):
    def __init__(self, pandasFrame, config=None):
        """
        Pandas_Time_Series_Tensor_Dataset prepared pandas data into sequence data type
        
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
        self.df = pandasFrame
        self.ixDict= None
        if config is None:
            self.config = {}
        else:
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
        tensorType : {'sequence','label','category','same'}
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


    def expand_dataFrame_by_date(self, categoryCol,timeSeriesCol,newIX=True,byCategory=True,dropna=False):
        """
        expand_dataFrame_by_date A help function to prepare dataFrame for tensor transformation
        
        It will change the original dataFrame of
        byCategory is set to be True [x1,x2,x3,time series,category] -> [x1_t1,x1_t2...x3_t|index->category]
        byCategory is set to be False [x1,x2,x3,time series,category] -> [category_x1_t1,category_x1_t2...category_x3_t]
        
        Note: 
        x is column name 
        t represent time series column i.e. Date --> YYYY-MM-DD format is recommended
        
        Parameters
        ----------
        categoryCol : str
            column name of category
        timeSeriesCol : str
            column name of time series
        newIX : bool, optional
            if True, time series column will be converted into 1,2,3,....len(timeSeriesCol), by default True
        byCategory : bool, optional
            if True, the dataFrame will create new row instead of different column for categories, by default True
        dropna : bool, optional
            if True, nan column will be dropped, by default False
        
        Returns
        -------
        iterable
            the index of time series columns
        """
        if newIX:
            self.df = self.df.sort_values(timeSeriesCol,ascending = True)
            ixDict = dict(zip(self.df[timeSeriesCol].unique(),list(range(1,len(self.df[timeSeriesCol].unique())+1))))
            self.df[timeSeriesCol] = self.df[timeSeriesCol].apply(lambda x: ixDict[x])
        else:
            ixDict = self.df[timeSeriesCol].values

        if byCategory:
            self.df = self._pivot_df(self.df,categoryCol,timeSeriesCol,dropna)
        else:
            self.df = self._flatten_df(self.df,categoryCol,timeSeriesCol,dropna)
        self.ixDict = ixDict
        return self

    def _pivot_df(self,df,categoryCol,timeSeriesCol,dropna):
        df = df.pivot(categoryCol,timeSeriesCol,df.columns.drop([categoryCol,timeSeriesCol]))
        df.columns = list(map(lambda x: f"{x[0]}_{x[1]}",df.columns))
        if dropna:
            df = df.dropna(axis =1)
        return df

    def _flatten_df(self,df,categoryCol,timeSeriesCol,dropna):
        categoryList = df[categoryCol].unique()
        resDf = None
        for i in categoryList:
            subDf = df[df[categoryCol]==i]
            subDf = self._pivot_df(subDf,categoryCol,timeSeriesCol,dropna)
            subDf.columns = list(map(lambda x: f"{i}_{x}",subDf.columns))
            subDf = subDf.reset_index(drop=True)
            if resDf is None:
                resDf = subDf
            else:
                resDf = pd.concat([resDf,subDf],axis =1)
        return resDf


    def __repr__(self):
        return f"Tensor Transformer Config: {repr(self.config)}"



class Pandas_Time_Series_Panel_Dataset(object):

    def __init__(self,pandasFrame):
        """
        Pandas_Time_Series_Panel_Dataset prepares the dataset for traditional machine learning problem.
        
        It can convert the pandas Frame into multiple lagging features or create a lead feature as label.
        
        Parameters
        ----------
        pandasFrame : pandas dataFrame
            the dataFrame for preprocessing
        """
        self.df = pandasFrame

    def expand_dataFrame_by_category(self,indexCol,keyCol):
        """
        expand_dataFrame_by_category to create columns for different categories
        
        it convert the dataFrame from [x1,x2,x3,...,category] -> [category_x1,category_x2,category_x3,....]
        
        Parameters
        ----------
        indexCol : str
            the time series column
        keyCol : str
            the category column
        
        """
        keys = self.df[keyCol].unique()
        tmpDf = None
        for ix,k in enumerate(keys):
            if ix == 0:
                tmpDf = self.df[self.df[keyCol]==k]
                tmpDf = tmpDf.set_index(indexCol)
                tmpDf = tmpDf.drop(keyCol,axis=1)
                tmpDf.columns = list(map(lambda x: f'{x}_{k}',tmpDf.columns))
            else:
                df2 = self.df[self.df[keyCol]==k]
                df2 = df2.drop(keyCol,axis=1).set_index(indexCol)
                df2.columns = list(map(lambda x: f'{x}_{k}',df2.columns))
                tmpDf = tmpDf.join(df2,how='outer')
        self.df = pd.DataFrame(tmpDf.to_records())
        return self


    def make_slide_window(self,indexCol,windowSize,colList=None,groupby=None):
        """
        make_slide_window make lag features given with the range of window size
        
        this function will create lag features along with the given window size.
        if colList set to be None, all column will be used to create lag features.
        groupby is for category columns. This paramemter for dataFrame which did not
        expand by categories.
        
        Parameters
        ----------
        indexCol : str
            time series column
        windowSize : int
            lag number created
        colList : list, optional
            the columns used to create lag features, by default None
        groupby : str, optional
            category column, by default None

        """
        if colList is None:
            self.df = self.df.sort_values(indexCol,ascending = True)
            colList = self.df.columns.drop(indexCol).tolist()
        for col in colList:
            for i in range(1,windowSize+1):
                if groupby is None:
                    self.df[f'{col}_lag{str(i)}'] = self.df[col].shift(i)
                else:
                    if col == groupby:
                        continue
                    self.df[f'{col}_lag{str(i)}'] = self.df.groupby(groupby)[col].shift(i)
        return self


    def make_lead_column(self,indexCol,baseCol,leadNum,groupby=None):
        """
        make_lead_column this function will create lead feature along with the lead number
        
        this function is for creating label for supervised learning
        groupby is for category columns. This paramemter for dataFrame which did not
        expand by categories.

        Parameters
        ----------
        indexCol : str
            time series column
        baseCol : str
            the column for lead feature
        leadNum : int
            the lead time unit
        groupby : str, optional
            category column, by default None
        
        Returns
        -------
        [type]
            [description]
        """
        self.df = self.df.sort_values(indexCol,ascending = False)
        if groupby is None:
            self.df[f'{baseCol}_lead{str(leadNum)}'] = self.df[baseCol].shift(leadNum)
        else:
            self.df[f'{baseCol}_lead{str(leadNum)}'] = self.df.groupby(groupby)[baseCol].shift(leadNum)            
        return self

    def transform_dataFrame(self,colName,targetCol,timeSeriesCol,groupby,transformFunc,*args,**kwargs):
        """
        transform_dataFrame this function use apply method to transfrom dataFrame
        Note: the inpupt and output of transformFunc must be list or numpy array
        
        Parameters
        ----------
        colName : str
            target column for transformation
        targetCol : str
            the column to store new data
        timeSeriesCol : str
            time series column for sorting before apply function
        groupby: str
            the category column used for grouping data during transformation
            if this column is None, no grouping will be applied
        transformFunc : func
            the function implmented in the apply function
        axis : int, optional
            0 for row 1 for column, by default 1

        """
        if groupby is not None:
            manipulateList = []
            self.df = self.df.sort_values([groupby,timeSeriesCol])
            for i in self.df[groupby].unique():
                manipulateList.extend(transformFunc(self.df[self.df[groupby]==i][colName].values,*args,**kwargs))
            self.df[targetCol] = manipulateList
        else:
            self.df = self.df.sort_values(timeSeriesCol,ascending = True)
            self.df[targetCol] = transformFunc(self.df[colName],*args,**kwargs)
        return self

    def __repr__(self):
        return repr(self.df)
        