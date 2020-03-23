class feature_engineering(object):
    def __init__ (self,df,dimList,encoder = LabelEncoder):
        super().__init__()
        self._df = df
        self._dimList = dimList
        self.arr = df.drop(dimList,axis =1).values
        self.indexList = df.index.tolist()
        self._encoder = encoder
        self.encodeDict = None

    def _rolling_window(self,a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def _get_time_tensor(self,arr,window_size):
        tmp = self._rolling_window(arr,window_size+1)
        Xtensor = tmp[:,:-1]
        Ytensor = tmp[:,-1]
        return (Xtensor.reshape(-1,window_size,1),Ytensor.reshape(-1,1))

    def np_to_time_tensor_generator(self,windowSize):
        if np.ndim(self.arr) > 1:
            for ix,v in enumerate(arr):
                yield self._get_time_tensor(v,windowSize)
        else:
            yield self._get_time_tensor(self.arr,windowSize) 

    def _get_item_id(self,fullIndex):
        tmp = fullIndex.split('_')
        return '_'.join(tmp[:3])

    def _get_store_id(self,fullIndex):
        tmp = fullIndex.split('_')
        return '_'.join(tmp[3:5])

    def _get_cate_info(self,fullIndex,cateInfoDir):
        item_id = _get_item_id(fullIndex)
        store_id = _get_store_id(fullIndex)
        return pd.read_parquet(cateInfoDir,filters = [("item_id",'=',str(item_id)),("store_id",'=',str(store_id))])

    def _get_events(self,calendar_dir,sdate,edate):
        df = pd.read_csv(calendar_dir)
        df['d_num'] = df.d.apply(lambda x: x.replace('d_','')).astype('int')
        return df[df.d_num.apply(lambda x: x <= edate and x >= sdate)]

    def _get_future_events(self,calendar_dir, sdate,duration):
        edate = sdate+duration-1
        return _get_events(calendar_dir,sdate,edate)

    def _label_encode(self,arr):
        encoder = self._encoder
        enc_arr = encoder().fit_transform(arr)
        return enc_arr,encoder

    def pandas_to_categorical_encode(self):
        encodeDict = {}
        labelDict = {}
        for i in self._dimList:
            enc_arr,encoder = self._label_encode(self._df[i])
            encodeDict[i] = encoder
            labelDict[i] = enc_arr
        self.encodeDict = encodeDict
        return labelDict