Stock Forecast Using Single Denoised Data
=========================================

This example is going to demonstrate how to use time_seris_transform to
denoise data prepare lag/lead features. In this exmple, there are four
steps 1. prepare raw and denoise data 2. make lag/lead feature 3.
Sklearn Model & Visualize Test

Prepare raw data
----------------

In this example, we use Stock_Extractor to get an single stock data from
yahoo finance. Subsequently, we directly denoise the data using
make_techinical_indicator funciton. This function is a wrapper which can
take function object and transform the traget column.In
time_series_transform.transform_core_api.util, there are different
pre-made fucntion for feature engineering such as geometric moving
average, wavelet transformation, or fast fourier transformation.

.. code:: ipython3

    import pandas as pd
    import seaborn as sns
    import time_series_transform as tst
    from time_series_transform.transform_core_api.util import wavelet_denoising

.. code:: ipython3

    stock = tst.Stock_Extractor('googl','yahoo').get_stock_period('5y') # get period of stock
    df = stock.make_technical_indicator("Close","Close_Wavelet",wavelet_denoising,wavelet='haar').dataFrame # denoise data
    stock.plot(["Close","Close_Wavelet"])



.. image:: _static/output_3_0.png


Make lag/lead features
----------------------

After we prepared the data, we can use Pandas_Time_Series_Panel_Dataset
to make lag or lead data as feature or target values for machine
learning models. In this example, we only make 30-day lags data for
Close data with Wavelet transformation, and only 1 step forward data as
label.

.. code:: ipython3

    panel_trans = tst.Pandas_Time_Series_Panel_Dataset(df)
    panel_trans = panel_trans.make_slide_window('Date',30,['Close_Wavelet'])
    panel_trans = panel_trans.make_lead_column('Date','Close',1)
    df = panel_trans.df

.. code:: ipython3

    df = df.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends',
           'Stock Splits', 'Close_Wavelet', 'symbol'],axis =1).dropna()
    df = df.sort_values('Date')
    print(df.head())


.. parsed-literal::

             Date  Close_Wavelet_lag1  Close_Wavelet_lag2  Close_Wavelet_lag3  \
    0  2015-08-12          678.338125          678.338125          678.338125   
    1  2015-08-13          678.338125          678.338125          678.338125   
    2  2015-08-14          678.338125          678.338125          678.338125   
    3  2015-08-17          678.338125          678.338125          678.338125   
    4  2015-08-18          678.338125          678.338125          678.338125   
    
       Close_Wavelet_lag4  Close_Wavelet_lag5  Close_Wavelet_lag6  \
    0          678.338125          678.338125          678.338125   
    1          678.338125          678.338125          678.338125   
    2          678.338125          678.338125          613.123125   
    3          678.338125          613.123125          613.123125   
    4          613.123125          613.123125          661.683125   
    
       Close_Wavelet_lag7  Close_Wavelet_lag8  Close_Wavelet_lag9  ...  \
    0          678.338125          613.123125          613.123125  ...   
    1          613.123125          613.123125          661.683125  ...   
    2          613.123125          661.683125          661.683125  ...   
    3          661.683125          661.683125          637.403125  ...   
    4          661.683125          637.403125          637.403125  ...   
    
       Close_Wavelet_lag22  Close_Wavelet_lag23  Close_Wavelet_lag24  \
    0           657.870625           657.870625           657.870625   
    1           657.870625           657.870625           657.870625   
    2           657.870625           657.870625           657.870625   
    3           657.870625           657.870625           657.870625   
    4           657.870625           657.870625           657.870625   
    
       Close_Wavelet_lag25  Close_Wavelet_lag26  Close_Wavelet_lag27  \
    0           657.870625           657.870625           657.870625   
    1           657.870625           657.870625           657.870625   
    2           657.870625           657.870625           657.870625   
    3           657.870625           657.870625           657.870625   
    4           657.870625           657.870625           657.870625   
    
       Close_Wavelet_lag28  Close_Wavelet_lag29  Close_Wavelet_lag30  Close_lead1  
    0           657.870625           657.870625           657.870625       686.51  
    1           657.870625           657.870625           657.870625       689.37  
    2           657.870625           657.870625           631.807500       694.11  
    3           657.870625           631.807500           631.807500       688.73  
    4           631.807500           631.807500           631.807500       694.04  
    
    [5 rows x 32 columns]
    

Machine learning model & visualize test results
-----------------------------------------------

After we pre-process the data, we can use sklearn to predict the value.
In this example, we use Random Forest with Random Search Tuner to
prepare our model.

.. code:: ipython3

    from sklearn.model_selection import TimeSeriesSplit,RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor

.. code:: ipython3

    train = df[df.Date<'2020-07-01']
    test = df[df.Date >= '2020-07-01']

.. code:: ipython3

    rf = RandomForestRegressor(n_jobs =1,)
    haparms = [{
        "n_estimators":np.arange(100,200,10,dtype=int),
        "min_samples_split":np.arange(2,40,1,dtype=int),
        "min_samples_leaf":np.arange(2,40,1,dtype=int),
    }]
    
    randomSearch = RandomizedSearchCV(
        rf,
        haparms,5,
        cv = TimeSeriesSplit(3),
        n_jobs =1
    )

.. code:: ipython3

    trainX= train.drop(['Date','Close_lead1'],axis = 1)
    trainY = train.Close_lead1
    randomSearch.fit(trainX,trainY)




.. parsed-literal::

    RandomizedSearchCV(cv=TimeSeriesSplit(max_train_size=None, n_splits=3),
                       error_score=nan,
                       estimator=RandomForestRegressor(bootstrap=True,
                                                       ccp_alpha=0.0,
                                                       criterion='mse',
                                                       max_depth=None,
                                                       max_features='auto',
                                                       max_leaf_nodes=None,
                                                       max_samples=None,
                                                       min_impurity_decrease=0.0,
                                                       min_impurity_split=None,
                                                       min_samples_leaf=1,
                                                       min_samples_split=2,
                                                       min_weight_fraction_leaf=0...
           19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
           36, 37, 38, 39]),
                                             'min_samples_split': array([ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
           19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
           36, 37, 38, 39]),
                                             'n_estimators': array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])}],
                       pre_dispatch='2*n_jobs', random_state=None, refit=True,
                       return_train_score=False, scoring=None, verbose=0)



.. code:: ipython3

    test= test.sort_values('Date')
    testX= test.drop(['Date','Close_lead1'],axis = 1)
    testY = test.Close_lead1
    test_prd = randomSearch.predict(testX)
    train_prd = randomSearch.predict(trainX)

.. code:: ipython3

    compareFrame = pd.DataFrame({'train_prd':train_prd,'train_real':train.Close_lead1.tolist()},index = train.Date)
    compareFrame_test = pd.DataFrame({'test_prd':test_prd,'test_real':test.Close_lead1.tolist()},index = test.Date)
    compareFrame = compareFrame.append(compareFrame_test)
    compareFrame['Date'] = compareFrame.index
    compareFrame.index = list(range(1,len(compareFrame)+1))

.. code:: ipython3

    sns.lineplot(data= [
        compareFrame[compareFrame.Date >'2020-04-01'].train_real, 
        compareFrame[compareFrame.Date >='2020-07-01'].test_real,
        compareFrame[compareFrame.Date >'2020-04-01'].train_prd, 
        compareFrame[compareFrame.Date >='2020-07-01'].test_prd, 
    ])




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x120e78f0048>




.. image:: _static/output_14_1.png


Stock Forecast Using Multiple Stock Features
============================================

This example is going to use a set of stocks to predict one stock. This
example also implments different algorithms for feature engineering.
Furthermore, this example will show how to implment functional object in
Pandas_Time_Series_Panel_Dataset class. There are three steps in this
example 1. prepare stocks data through Portfolio_Extractor api and
transform data 2. making lag/lead data and other feature engineering
function using Pandas_Time_Series_Panel_Dataset 3. sklearn model and
visualize test results

Prepare stocks data and transformation
--------------------------------------

Portfolio_Extractor takes a list of stock ticks, and return an Portfolio
objects. In this example, we prepare 5 year of data. Subsequently, we
transform Close, Open, High, Low values using wavelet, fast fourier, and
geometric moving average.

.. code:: ipython3

    import time_series_transform as tst
    from time_series_transform.transform_core_api.util import wavelet_denoising,rfft_transform,geometric_ma

.. code:: ipython3

    port = tst.Portfolio_Extractor(['googl','aapl','fb'],'yahoo').get_portfolio_period('5y')

.. code:: ipython3

    colList = ['Close','Open','High','Low']
    denoiseCol = []
    for col in colList:
        port.make_technical_indicator(col,f'{col}_wavelet',wavelet_denoising,wavelet='haar')
        port.make_technical_indicator(col,f'{col}_fft',rfft_transform,threshold=0.001)
        port.make_technical_indicator(col,f'{col}_gma20',geometric_ma,windowSize=20)
        port.make_technical_indicator(col,f'{col}_gma200',geometric_ma,windowSize=200)
        denoiseCol.extend([f'{col}_wavelet',f'{col}_fft',f'{col}_gma200',f'{col}_gma20'])

.. code:: ipython3

    df = port.get_portfolio_dataFrame()
    print(df.head())


.. parsed-literal::

             Date    Open    High     Low   Close   Volume  Dividends  \
    0  2015-08-12  694.49  696.00  680.51  691.47  2924900        0.0   
    1  2015-08-13  689.20  694.03  682.18  686.51  1817700        0.0   
    2  2015-08-14  684.04  692.26  682.90  689.37  1379900        0.0   
    3  2015-08-17  688.04  694.74  683.06  694.11  1234000        0.0   
    4  2015-08-18  691.07  695.76  685.32  688.73  1385500        0.0   
    
       Stock Splits  Close_wavelet    Close_fft  ...  Open_gma200  High_wavelet  \
    0             0     678.338125  1027.446516  ...          NaN       692.755   
    1             0     678.338125  1027.446516  ...          NaN       692.755   
    2             0     678.338125  1027.446516  ...          NaN       692.755   
    3             0     678.338125  1027.446516  ...          NaN       692.755   
    4             0     678.338125  1027.446516  ...          NaN       692.755   
    
          High_fft  High_gma20  High_gma200  Low_wavelet      Low_fft  Low_gma20  \
    0  1036.619071         NaN          NaN   673.884375  1017.281849        NaN   
    1  1036.619071         NaN          NaN   673.884375  1017.281849        NaN   
    2  1036.619071         NaN          NaN   673.884375  1017.281849        NaN   
    3  1036.619071         NaN          NaN   673.884375  1017.281849        NaN   
    4  1036.619071         NaN          NaN   673.884375  1017.281849        NaN   
    
       Low_gma200  symbol  
    0         NaN   googl  
    1         NaN   googl  
    2         NaN   googl  
    3         NaN   googl  
    4         NaN   googl  
    
    [5 rows x 25 columns]
    

Lag/lead data and other feature engineering
-------------------------------------------

Here, we create an moving standard deviation feature and make 20-day-lag
with denoised data from previous step. In order to create the proper
feature, we have to be aware of its category. That means that we have to
group by each stocks. After making new feature, we can create lag data
and expand it by its category.

To manipulate the data using transform_dataFrame function, the first
argument of the function must be the data array/list, and the output
must be numpy array or list.

.. code:: ipython3

    df = df.drop(['Volume','Dividends','Stock Splits'],axis =1)
    panel_transform = tst.Pandas_Time_Series_Panel_Dataset(df)

.. code:: ipython3

    import pandas as pd
    def moving_std (arr,windowSize):
        res = pd.Series(arr).rolling(windowSize).std()
        return res
    panel_transform = panel_transform.transform_dataFrame('Close','Close_std','Date','symbol',moving_std,windowSize = 10)

.. code:: ipython3

    panel_transform = panel_transform.make_slide_window(
        indexCol = 'Date',
        windowSize = 20,
        colList = denoiseCol+['Close_std'],
        groupby = 'symbol')
    panel_transform = panel_transform.expand_dataFrame_by_category('Date','symbol')

.. code:: ipython3

    print(panel_transform.df.head())


.. parsed-literal::

             Date  Open_aapl  High_aapl  Low_aapl  Close_aapl  Close_wavelet_aapl  \
    0  2015-08-12     103.99     106.66    101.31      106.49          105.099219   
    1  2015-08-13     107.23     107.57    105.85      106.41          105.099219   
    2  2015-08-14     105.64     107.48    105.36      107.16          105.099219   
    3  2015-08-17     107.23     108.72    106.74      108.27          105.099219   
    4  2015-08-18     107.59     108.53    107.21      107.66          105.099219   
    
       Close_fft_aapl  Close_gma20_aapl  Close_gma200_aapl  Open_wavelet_aapl  \
    0       173.68827               NaN                NaN         107.643125   
    1       173.68827               NaN                NaN         107.643125   
    2       173.68827               NaN                NaN         107.643125   
    3       173.68827               NaN                NaN         107.643125   
    4       173.68827               NaN                NaN         107.643125   
    
       ...  Close_std_lag11_googl  Close_std_lag12_googl  Close_std_lag13_googl  \
    0  ...                    NaN                    NaN                    NaN   
    1  ...                    NaN                    NaN                    NaN   
    2  ...                    NaN                    NaN                    NaN   
    3  ...                    NaN                    NaN                    NaN   
    4  ...                    NaN                    NaN                    NaN   
    
       Close_std_lag14_googl  Close_std_lag15_googl  Close_std_lag16_googl  \
    0                    NaN                    NaN                    NaN   
    1                    NaN                    NaN                    NaN   
    2                    NaN                    NaN                    NaN   
    3                    NaN                    NaN                    NaN   
    4                    NaN                    NaN                    NaN   
    
       Close_std_lag17_googl  Close_std_lag18_googl  Close_std_lag19_googl  \
    0                    NaN                    NaN                    NaN   
    1                    NaN                    NaN                    NaN   
    2                    NaN                    NaN                    NaN   
    3                    NaN                    NaN                    NaN   
    4                    NaN                    NaN                    NaN   
    
       Close_std_lag20_googl  
    0                    NaN  
    1                    NaN  
    2                    NaN  
    3                    NaN  
    4                    NaN  
    
    [5 rows x 1084 columns]
    

.. code:: ipython3

    panel_transform = panel_transform.make_lead_column('Date','Close_googl',1)
    df = panel_transform.df
    df = df.dropna()

.. code:: ipython3

    from sklearn.model_selection import TimeSeriesSplit,RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    import seaborn as sns

.. code:: ipython3

    train = df[df.Date<'2020-07-01']
    test = df[df.Date >= '2020-07-01']

.. code:: ipython3

    rf = RandomForestRegressor(n_jobs =1)
    haparms = [{
        "n_estimators":np.arange(100,200,10,dtype=int),
        "min_samples_split":np.arange(2,40,1,dtype=int),
        "min_samples_leaf":np.arange(2,40,1,dtype=int),
    }]
    
    randomSearch = RandomizedSearchCV(
        rf,
        haparms,10,
        cv = TimeSeriesSplit(3),
        n_jobs =1
    )

.. code:: ipython3

    train = train.sort_values('Date')
    trainX= train.drop(['Date','Close_googl_lead1'],axis = 1)
    trainY = train.Close_googl_lead1
    randomSearch.fit(trainX,trainY)
    train_prd = randomSearch.predict(trainX)

.. code:: ipython3

    test= test.sort_values('Date')
    testX= test.drop(['Date','Close_googl_lead1'],axis = 1)
    testY = test.Close_googl_lead1
    test_prd = randomSearch.predict(testX)

.. code:: ipython3

    compareFrame = pd.DataFrame({'train_prd':train_prd,'train_real':train.Close_googl_lead1.tolist()},index = train.Date)
    compareFrame_test = pd.DataFrame({'test_prd':test_prd,'test_real':test.Close_googl_lead1.tolist()},index = test.Date)
    compareFrame = compareFrame.append(compareFrame_test)
    compareFrame['Date'] = compareFrame.index
    compareFrame.index = list(range(1,len(compareFrame)+1))

.. code:: ipython3

    sns.lineplot(data= [
        compareFrame[compareFrame.Date >'2020-04-01'].train_real, 
        compareFrame[compareFrame.Date >='2020-07-01'].test_real,
        compareFrame[compareFrame.Date >'2020-04-01'].train_prd, 
        compareFrame[compareFrame.Date >='2020-07-01'].test_prd, 
    ])




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1fa4b72c048>




.. image:: _static/output_32_1.png


