import copy
from time_series_transform.transform_core_api.base import (
    Time_Series_Data,
    Time_Series_Data_Collection
    )
import numpy as np

class io_base (object):
    def __init__(self,time_series,timeSeriesCol,mainCategoryCol):
        self.time_series = copy.deepcopy(time_series)
        self.timeSeriesCol = timeSeriesCol
        self.mainCategoryCol = mainCategoryCol

    def to_single(self,dictList,timeSeriesCol):
        tsd = Time_Series_Data()
        if timeSeriesCol is None:
            raise KeyError("time series index is required")
        tsd.set_time_index(dictList[timeSeriesCol],timeSeriesCol)
        for i in dictList:
            if i == timeSeriesCol:
                continue
            tsd.set_data(dictList[i],i)
        return tsd
    
    def to_collection(self,dictList,timeSeriesCol,mainCategoryCol):
        tsd = Time_Series_Data()
        if timeSeriesCol is None:
            raise KeyError("time series index is required")
        tsd.set_time_index(dictList[timeSeriesCol],timeSeriesCol)
        for i in dictList:
            if i == timeSeriesCol:
                continue
            tsd.set_data(dictList[i],i)
        tsc = Time_Series_Data_Collection(tsd,timeSeriesCol,mainCategoryCol)
        return tsc

    def from_collection(self,expandCategory,expandTimeIx,preprocessType='ignore'):
        transCollection = copy.deepcopy(self.time_series)
        if preprocessType == 'remove':
            transCollection = transCollection.remove_different_time_index()
        elif preprocessType == 'pad':
            transCollection = transCollection.pad_time_index()
        elif preprocessType != 'ignore':
            raise KeyError('preprocess type must be remove, pad, or ignore')
        if expandCategory:
            transCollection = self._expand_dict_category(transCollection)
        if expandTimeIx:
            transCollection = self._expand_dict_date(transCollection)
        res = {}
        for i in transCollection:
            if isinstance(transCollection[i],Time_Series_Data):
                data = transCollection[i][:]
            else:
                data = transCollection[i]
            categoryList = np.empty(transCollection[i].time_length)
            categoryList[:] = i
            data[self.mainCategoryCol] = categoryList
            for key in data:
                if key not in res:
                    res[key] = list(data[key])
                else:
                    res[key] += list(data[key])
        return res

    def from_single(self,expandTime):
        if expandTime:
            tmp = {"1":self.time_series}
            return self._expand_dict_date(tmp)['1']
        else:
            dfDict = {}
            dfDict.update(self.time_series.time_index)
            dfDict.update(self.time_series.labels)
            dfDict.update(self.time_series.data)
        return dfDict

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
                    if not isinstance(a[i][t],list) or not isinstance(a[i][t],np.ndarray):
                        tmp[f"{t}_{a[i][timeIx]}"]=[a[i][t]]
                    else:
                        tmp[f"{t}_{a[i][timeIx]}"]=a[i][t]
            dct[k] = tmp
        return dct

