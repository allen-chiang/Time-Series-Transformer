import copy
from time_series_transform.transform_core_api.base import (
    Time_Series_Data,
    Time_Series_Data_Collection
    )
import numpy as np

class io_base (object):
    def __init__(self,time_series,timeSeriesCol,mainCategoryCol):
        """
        io_base 
        IO class
        
        Parameters
        ----------
        time_series : Time_Series_Data or Time_Series_Data_Collection
            input data
        timeSeriesCol : str or int
            index of time period column
        mainCategoryCol : str of int
            index of category column
        """
        if isinstance(time_series,(Time_Series_Data,Time_Series_Data_Collection)):
            self.time_series = copy.deepcopy(time_series)
            self.dictList = None
        else:
            self.time_series = None
            self.dictList = copy.deepcopy(time_series)
        self.timeSeriesCol = timeSeriesCol
        self.mainCategoryCol = mainCategoryCol

    def to_single(self):
        """
        to_single transform data to Time_Series_Data
        
        Returns
        -------
        Time_Series_Data
        
        Raises
        ------
        KeyError
            invalid data
        """
        tsd = Time_Series_Data()
        if self.timeSeriesCol is None:
            raise KeyError("time series index is required")
        tsd.set_time_index(self.dictList[self.timeSeriesCol],self.timeSeriesCol)
        for i in self.dictList:
            if i == self.timeSeriesCol:
                continue
            tsd.set_data(self.dictList[i],i)
        return tsd
    
    def to_collection(self):
        """
        to_collection transform data into Time_Series_Data_Collection
        
        Returns
        -------
        Time_Series_Data_Collection
        
        Raises
        ------
        KeyError
            invalid input
        """
        if self.timeSeriesCol is None:
            raise KeyError("time series index is required")
        tsd = Time_Series_Data(self.dictList,self.timeSeriesCol)
        tsc = Time_Series_Data_Collection(tsd,self.timeSeriesCol,self.mainCategoryCol)
        return tsc

    def from_collection(self,expandCategory,expandTimeIx,preprocessType='ignore'):
        """
        from_collection prepare Time_Series_Data_Collection into dict of list
        
        Parameters
        ----------
        expandCategory : bool
            whether to expand category
        expandTime : bool
            whether to expand time
        preprocessType : ['ignore','pad','remove']
            preprocess data time across categories
        
        Returns
        -------
        dict of list
        
        Raises
        ------
        ValueError
            invalid data
        KeyError
            invalid key
        """
        transCollection = copy.deepcopy(self.time_series)
        transCollection =  transCollection.sort()
        if preprocessType == 'remove':
            transCollection = transCollection.remove_different_time_index()
        elif preprocessType == 'pad':
            transCollection = transCollection.pad_time_index()
        elif preprocessType == 'ignore':
            tmp = None
            diffTime = False
            for i in transCollection:
                if tmp is None:
                    tmp = transCollection[i].time_index[transCollection._time_series_Ix].tolist()
                    continue
                timeList = transCollection[i].time_index[transCollection._time_series_Ix].tolist()
                if set(tmp) != set(timeList):
                    diffTime = True
                tmp = timeList
                if diffTime and (False == (expandCategory == expandTimeIx)):
                    raise ValueError('category time length should be in consist. otherwise, use pad or remove pre-process type. ')
        else:
            raise KeyError('preprocess type must be remove, pad, or ignore')

        if expandCategory:
            transCollection = self._expand_dict_category(transCollection)
        if expandTimeIx:
            transCollection = self._expand_dict_date(transCollection)
        res = {}
        for i in transCollection:
            if isinstance(transCollection[i],Time_Series_Data):
                data = transCollection[i][:]
                catLen = transCollection[i].time_length
            else:
                data = transCollection[i]
                tmpKey =list(data.keys())[0]
                catLen = len(data[tmpKey])
            if not expandCategory:
                categoryList = [i for _ in range(catLen)]
                data[self.mainCategoryCol] = categoryList
            for key in data:
                if key not in res:
                    res[key] = list(data[key])
                else:
                    res[key] += list(data[key])
        return res

    def from_single(self,expandTime):
        """
        from_single transform Time_Series_Data into dict of list
        
        Parameters
        ----------
        expandTime : bool
            whether to expand Time
        
        Returns
        -------
        Time_Series_Data
        """
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

