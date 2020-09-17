import copy
import numpy as np
from time_series_transform.transform_core_api.base import (
    Time_Series_Data,
    Time_Series_Data_Colleciton
    )

class io_base (object):
    def __init__(self,time_series,timeSeriesCol,mainCategoryCol):
        self.time_series = copy.deepcopy(time_series)
        self.timeSeriesCol = timeSeriesCol
        self.mainCategoryCol = mainCategoryCol

    def from_single(self):
        pass
    
    def from_collection(self):
        pass

    def to_collection(self,expandCategory,expandTimeIx,remove):
        transCollection = copy.deepcopy(self.time_series)
        if remove:
            transCollection = transCollection.remove_different_time_index()
        else:
            transCollection = transCollection.pad_time_index()
        if expandCategory:
            transCollection = self._expand_dict_category(transCollection)
        if expandTimeIx:
            transCollection = self._expand_dict_date(transCollection)
        resList = []
        for i in transCollection:
            if isinstance(transCollection[i],Time_Series_Data):
                data = transCollection[i][:]
                for j in data:
                    data[j] = data[j].tolist()
            else:
                data = transCollection[i]
            resList.append(data)
        res = {}
        for i in resList:
            for key in i:
                if key not in res:
                    res[key] = i[key]
                else:
                    res[key]+= i[key]
        return res


    def to_single(self,expandTime):
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

