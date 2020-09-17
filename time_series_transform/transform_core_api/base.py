import copy
import numpy as np
import pandas as pd
import pprint
from joblib import Parallel, delayed
from collections import ChainMap
from collections import Counter

class Time_Series_Data(object):

    def __init__(self):
        self.time_length = 0
        self._data = {}
        self._time_index= {}
        self._labels = {}

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def time_index(self):
        return self._time_index

    def set_data(self,inputData,label):
        if len(inputData) != self.time_length:
            raise ValueError('input data has different time length')
        self._data[label] = np.array(inputData)
        return self


    def set_labels(self,inputData,label):
        if len(inputData) != self.time_length:
            raise ValueError('input data has different time length')
        self._labels[label] = np.array(inputData)
        return self

    def remove(self,key):
        if key in self.data:
            self._data.pop(key)
        if key in self.labels:
            self._labels.pop(key)
        return self


    def set_time_index(self,inputData,label):
        self._time_index = {}
        self._time_index[label] = np.array(inputData)
        self.time_length = len(inputData)
        return self

    def _get_dictionary_list_info(self,dictionary,indexSlice,label):
        res = {}
        if label is None:
            for i in dictionary:
                res[i] = dictionary[i][indexSlice]
        else:
            res[label] = dictionary[label][indexSlice]
        return res

    def __getitem__(self,ix):
        tmpInfo = self.labels
        tmpInfo.update(self.data)
        info = {}
        if isinstance(ix,tuple):
            t = ix[0]
            info.update(self._get_dictionary_list_info(self.time_index,t,None))
            for q in ix[1]:
                info.update(self._get_dictionary_list_info(tmpInfo,t,q))
        else:
            info.update(self._get_dictionary_list_info(self.time_index,ix,None))
            info.update(self._get_dictionary_list_info(tmpInfo,ix,None))
        return info

    def _reorder_list(self,sortingList,targetList,ascending):
        descending = 1-ascending
        ixList = sorted(range(len(sortingList)), key=lambda k: sortingList[k],reverse = descending)
        ordered_list = [targetList[i] for i in ixList]
        return np.array(ordered_list)

    def sort(self,ascending=True):
        sortingList = list(self.time_index.values())[0]
        for data in self.data:
            self.data[data] = self._reorder_list(sortingList,self.data[data],ascending)
        for label in self.labels:
            self.labels[label] = self._reorder_list(sortingList,self.labels[label],ascending)
        for time in self.time_index:
            self.time_index[time] = self._reorder_list(sortingList,self.time_index[time],ascending)
        return self

    def make_dataframe(self):
        dfDict = {}
        dfDict.update(self.time_index)
        dfDict.update(self.labels)
        dfDict.update(self.data)
        for i in dfDict:
            dfDict[i] = dfDict[i].tolist()
        return pd.DataFrame(dfDict)

    def _single_transform(self,colName,func,*args,**kwargs):
        if colName in self.data:
            arr = self.data[colName]
            return func(arr,*args,**kwargs),'data'
        else:
            arr = self.labels[colName]
            return func(arr,*args,**kwargs),'labels'

    def _list_transform(self,inputList,func,*args,**kwargs):
        arrDict = {}
        outputType = 'label'
        for col in inputList:
            if col in self.data:
                arrDict[col] = self.data[col]
                outputType='data'
            else:
                arrDict[col] = self.labels[col]
        arrDict = func(arrDict,*args,**kwargs)
        return arrDict,outputType

    def transform(self,inputLabels,newName,func,*args,**kwargs):
        # transform
        if isinstance(inputLabels,list):
            arr,outputType = self._list_transform(inputLabels,func,*args,**kwargs)
        else:
            arr,outputType = self._single_transform(inputLabels,func,*args,**kwargs)

        # organize into dict
        if isinstance(arr,pd.DataFrame):
            arr = arr.to_dict(orient='list')
            arr = { f"{newName}_{k}": v for k, v in arr.items() }
        elif isinstance(arr,list) or isinstance(arr,np.ndarray):
            arr = {newName:np.array(arr)}   
        elif isinstance(arr,pd.Series):
            arr = {newName:arr.values}

        if outputType == 'data':
            self._data.update(arr)
        else:
            self._labels.update(arr)
        # update existing dict
        return self
        

    def _get_all_info(self):
        dfDict = {}
        dfDict.update(self.time_index)
        dfDict.update(self.labels)
        dfDict.update(self.data)
        return dfDict

    def __repr__(self):
        return str(self._get_all_info())

    def __eq__(self, other):
        left = self._get_all_info()
        right = other._get_all_info()
        if len(left) != len(right):
            return False
        for i in left:
            if i not in right:
                return False
            left[i] = list(left[i])
            right[i] = list(right[i])
        return left == right

        
class Time_Series_Data_Colleciton(object):
    def __init__(self,time_series_data,time_seriesIx,categoryIx):
        super().__init__()
        if isinstance(time_series_data,dict):
            if self._check_dict_type(time_series_data):
                self._time_series_data_collection = time_series_data
        else:
            self._time_series_data_collection = self._expand_time_series_data(time_series_data,categoryIx)
        self._time_series_Ix = time_seriesIx
        self._categoryIx = categoryIx

    def _check_dict_type(self,time_series_data):
        check = True
        for i in time_series_data:
            check = isinstance(time_series_data[i],Time_Series_Data)
            if check == False:
                return check
        return check

    @property
    def time_series_data_collection(self):
        return self._time_series_data_collection

    def set_time_series_data_collection(self,ix,time_series_data):
        if isinstance(time_series_data,Time_Series_Data):
            self._time_series_data_collection[ix] = time_series_data
        else:
            raise ValueError("data must be Time_Series_Data type")


    def remove(self,key):
        if key in self._time_series_data_collection:
            self._time_series_data_collection.pop(key)
        return self

    def _expand_time_series_data(self,time_series_data,categoryIx):
        dct = {}
        for i in list(set(time_series_data[:,[categoryIx]][categoryIx])):
            ixList = np.where(time_series_data[:,[categoryIx]][categoryIx]==i)
            tmp = {}
            tmp = Time_Series_Data()
            for t in time_series_data.time_index:
                tmp.set_time_index(time_series_data.time_index[t][ixList],t)
            for d in time_series_data.data:
                tmp.set_data(time_series_data.data[d][ixList],d)
            for l in time_series_data.labels:
                if l == categoryIx:
                    continue
                tmp.set_data(time_series_data.labels[l][ixList],l)
            dct[i] = tmp
        return dct


    def _parallel_transform(self,category,time_series_data,inputLabels,newName,func,*args,**kwargs):
        return {category:time_series_data.transform(inputLabels,newName,func,*args,**kwargs)}


    def transform(self,inputLabels,newName,func,n_jobs =1,verbose = 0,backend='loky',*args,**kwargs):
        dctList= Parallel(n_jobs=n_jobs,verbose = verbose, backend=backend)(delayed(self._parallel_transform)(
            c,self._time_series_data_collection[c],inputLabels,newName,func,*args,**kwargs
            ) for c in self.time_series_data_collection)
        self._time_series_data_collection = dict(ChainMap(*dctList))
        return self

    def remove_different_time_index(self):
        timeix = []
        for i in self._time_series_data_collection:
            timeix.extend(self._time_series_data_collection[i][:][self._time_series_Ix])
        timeix = Counter(timeix)
        timeCol = [k for k,v in timeix.items() if v == len(self._time_series_data_collection)]    
        for i in self._time_series_data_collection:
            tmp_time = Time_Series_Data()
            ix = np.isin(self._time_series_data_collection[i][:][self._time_series_Ix],timeCol)
            for t in self._time_series_data_collection[i].time_index:
                tmp = self._time_series_data_collection[i].time_index[t][ix]
                tmp_time.set_time_index(tmp,t)
            for d in self._time_series_data_collection[i].data:
                tmp = self._time_series_data_collection[i].data[d][ix]
                tmp_time.set_data(tmp,d)
            for l in self._time_series_data_collection[i].labels:
                tmp = self._time_series_data_collection[i].labels[l][ix]
                tmp_time.set_labels(tmp,l)               
            self._time_series_data_collection[i] = tmp_time
        return self

    def pad_time_index(self,fillMissing=np.nan):
        timeix = []
        for i in self._time_series_data_collection:
            timeix.extend(self._time_series_data_collection[i][:][self._time_series_Ix]) 
        timeix = sorted(list(set(timeix)))
        for i in self._time_series_data_collection:
            tmp_time = Time_Series_Data()
            tmp_time.set_time_index(timeix,self._time_series_Ix)
            tmp = self._time_series_data_collection[i]
            for t in tmp.time_index:
                posList= np.isin(timeix,tmp.time_index[t])
            for d in tmp.data:
                nanList = np.empty(len(timeix))
                nanList[:] = fillMissing
                nanList[posList] = tmp.data[d]
                tmp_time.set_data(nanList,d)
            for l in tmp.labels:
                nanList = np.empty(len(timeix))
                nanList[:] = fillMissing
                nanList[posList] = tmp.labels[l]
                tmp_time.set_labels(nanList,l)
            self._time_series_data_collection[i] = tmp_time
        return self

    def sort(self,ascending=True,categoryList=None):
        if categoryList is None:
            categoryList = list(self._time_series_data_collection.keys())
        for i in categoryList:
            self._time_series_data_collection[i] =self._time_series_data_collection[i].sort(ascending)
        return self


    def __iter__(self):
        for i in self.time_series_data_collection:
            yield i

    def __repr__(self):
        return str(self._time_series_data_collection)

    def __getitem__(self,ix):
        return self._time_series_data_collection[ix]

    def _expand_dict_category(self,collectionDict):
        time_series = Time_Series_Data()
        for i in collectionDict:
            tmp =collectionDict[i]
            tmp.sort()
            for t in tmp.time_index:
                time_series.set_time_index(tmp.time_index[t],t)
            for d in tmp.data:
                time_series.set_data(tmp.data[d],f'{d}_{i}')
            for l in tmp.labels:
                time_series.set_labels(tmp.labels[l],f'{l}_{i}')
            
        return {'1':time_series}

    def _expand_dict_date(self,collectionDict):
        dct = {}
        for k in collectionDict:
            tmp = {}
            a = collectionDict[k]
            for i in range(a.time_length):
                timeIx = list(a.time_index.keys())[0]
                for t in a[i]:
                    if t in a.time_index:
                        continue
                    if  not isinstance(a[i][t],list) or not isinstance(a[i][t],np.ndarray):
                        tmp[f"{t}_{a[i][timeIx]}"]=[a[i][t]]
                    else:
                        tmp[f"{t}_{a[i][timeIx]}"]=a[i][t]
            dct[k] = tmp
        return dct


    def make_dataframe(self,expandCategory,expandTimeIx):
        resDf = pd.DataFrame()
        transCollection = copy.copy(self.time_series_data_collection)
        if expandCategory:
            transCollection = self._expand_dict_category(transCollection)
        if expandTimeIx:
            transCollection = self._expand_dict_date(transCollection)
        for i in transCollection:
            if expandTimeIx == False:
                data = transCollection[i][:]
                for j in data:
                    data[j] = data[j].tolist()
                tmp = pd.DataFrame(data)
            else:
                data = transCollection[i]
                for j in data:
                    data[j] = data[j].tolist()
                tmp = pd.DataFrame(data)
            if expandCategory == False:
                tmp[self._categoryIx] = i
            resDf = resDf.append(tmp)
        return resDf