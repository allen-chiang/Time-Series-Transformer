import numpy as np
from time_series_transform.io.base import io_base
from time_series_transform.transform_core_api.base import (
    Time_Series_Data_Collection,
    Time_Series_Data
    )


class Numpy_IO (io_base):
    def __init__(self, time_series, timeSeriesColIx, mainCategoryColIx):
        super().__init__(time_series, timeSeriesColIx, mainCategoryColIx)
        if self.dictList is not None:
            self.dictList = {}
            for i in range(len(time_series.T)):
                self.dictList[i] = time_series.T[i]

    def from_numpy (self):
        if self.mainCategoryCol is None:
            return self.to_single()
        return self.to_collection()


    def to_numpy(self,expandTime,expandCategory,preprocessType):
        if isinstance(self.time_series,Time_Series_Data):
            data = self.from_single(expandTime)
        if isinstance(self.time_series,Time_Series_Data_Collection):
            data = self.from_collection(expandCategory,expandTime,preprocessType)
        return np.asarray(list(data.values())).T


def from_numpy(numpyArray,timeSeriesCol,mainCategoryCol=None):
    if not isinstance(numpyArray,np.ndarray):
        raise ValueError('input data must be numpy array')
    numpyio = Numpy_IO(numpyArray,timeSeriesCol,mainCategoryCol)
    return numpyio.from_numpy()

def to_numpy(time_series_data,expandCategory,expandTime,preprocessType,seperateLabels=False):
    labelsList = []
    if isinstance(time_series_data,Time_Series_Data):
        numpyio = Numpy_IO(time_series_data,time_series_data.time_seriesIx,None)
        expandTime = None
        expandCategory = None
        labelsList = list(time_series_data.labels.keys())
    elif isinstance(time_series_data,Time_Series_Data_Collection):
        numpyio = Numpy_IO(
            time_series_data,
            time_series_data._time_series_Ix,
            time_series_data._categoryIx
            )
        for i in time_series_data:
            labelsList.extend(list(time_series_data[i].labels.keys()))
            labelList = list(set(labelsList))
    else:
        raise ValueError('input data should be time_series_data or time_series_collection')
    npList = numpyio.to_numpy(expandTime,expandCategory,preprocessType)
    if seperateLabels == False:
        return npList
    npShape = np.arange(npList.shape[1])
    normalIx = np.isin(npShape,labelsList,invert = True)
    return npList[:,npShape[normalIx]], npList[:,labelsList]


__all__= [
    'from_numpy',
    'to_numpy'
]