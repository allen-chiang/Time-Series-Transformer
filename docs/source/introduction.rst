.. code:: ipython3

    import pandas as pd
    import numpy as np
    from time_series_transform.sklearn import *
    import time_series_transform as tst

Introduction
============

This package provides tools for time series data preprocessing. There
are two main components inside the package: Time_Series_Transformer and
Stock_Transformer. Time_Series_Transformer is a general class for all
type of time series data, while Stock_Transformer is a sub-class of
Time_Series_Transformer. Time_Series_Transformer has different functions
for data manipulation, io transformation, and making simple plots. This
tutorial will take a quick look at the functions for data manipulation
and basic io. For the plot functions, there will be other tutorial to
explain.

Time_Series_Transformer
=======================

Since all the time series data having time data, Time_Series_Transformer
is required to specify time index. The basic time series data is time
series data with no special category. However, there a lot of cases that
a time series data is associating with categories. For example,
inventory data is usually associate with product name or stores, or
stock data is having different ticker names or brokers. To address this
question, Time_Series_Transformer can specify the main category index.
Given the main category index, the data can be manipulated in parallel
corresponding to its category.

Here is a simple example to create a Time_Series_Transformer without
specifying its category.

.. code:: ipython3

    data = {
        'time':[1,2,3,4,5],
        'data1':[1,2,3,4,5],
        'data2':[6,7,8,9,10]
    }
    trans = tst.Time_Series_Transformer(data,timeSeriesCol='time')
    trans




.. parsed-literal::

    data column
    -----------
    time
    data1
    data2
    time length: 5
    category: None
    



There are two ways to manipulate the data. The first way is use the
pre-made functions, and the second way is to use the transform function
and provide your custom function. There are six pre-made functions
including make_lag, make_lead, make_lag_sequence, make_lead_sequence,
and make_stack_sequence. In the following demonstration, we will show
each of the pre-made functions.

Pre-made functions
~~~~~~~~~~~~~~~~~~

make_lag and make_lead functions are going to create lag/lead data for
input columns. This type of manipulation could be useful for machine
learning.

.. code:: ipython3

    trans = tst.Time_Series_Transformer(data,timeSeriesCol='time')
    trans = trans.make_lag(
        inputLabels = ['data1','data2'],
        lagNum = 1,
        suffix = '_lag_',
        fillMissing = np.nan
                )
    print(trans.to_pandas())


.. parsed-literal::

       time  data1  data2  data1_lag_1  data2_lag_1
    0     1      1      6          NaN          NaN
    1     2      2      7          1.0          6.0
    2     3      3      8          2.0          7.0
    3     4      4      9          3.0          8.0
    4     5      5     10          4.0          9.0
    

.. code:: ipython3

    trans = tst.Time_Series_Transformer(data,timeSeriesCol='time')
    trans = trans.make_lead(
        inputLabels = ['data1','data2'],
        leadNum = 1,
        suffix = '_lead_',
        fillMissing = np.nan
                )
    print(trans.to_pandas())


.. parsed-literal::

       time  data1  data2  data1_lead_1  data2_lead_1
    0     1      1      6           2.0           7.0
    1     2      2      7           3.0           8.0
    2     3      3      8           4.0           9.0
    3     4      4      9           5.0          10.0
    4     5      5     10           NaN           NaN
    

make_lag_sequence and make_lead_sequence is to create a sequence for a
given window length and lag or lead number. This manipulation could be
useful for Deep learning.

.. code:: ipython3

    trans = tst.Time_Series_Transformer(data,timeSeriesCol='time')
    trans = trans.make_lag_sequence(
        inputLabels = ['data1','data2'],
        windowSize = 2,
        lagNum =1,
        suffix = '_lag_seq_'
    )
    print(trans.to_pandas())


.. parsed-literal::

       time  data1  data2 data1_lag_seq_2 data2_lag_seq_2
    0     1      1      6      [nan, nan]      [nan, nan]
    1     2      2      7      [nan, 1.0]      [nan, 6.0]
    2     3      3      8      [1.0, 2.0]      [6.0, 7.0]
    3     4      4      9      [2.0, 3.0]      [7.0, 8.0]
    4     5      5     10      [3.0, 4.0]      [8.0, 9.0]
    

.. code:: ipython3

    trans = tst.Time_Series_Transformer(data,timeSeriesCol='time')
    trans = trans.make_lead_sequence(
        inputLabels = ['data1','data2'],
        windowSize = 2,
        leadNum =1,
        suffix = '_lead_seq_'
    )
    print(trans.to_pandas())


.. parsed-literal::

       time  data1  data2 data1_lead_seq_2 data2_lead_seq_2
    0     1      1      6       [2.0, 3.0]       [7.0, 8.0]
    1     2      2      7       [3.0, 4.0]       [8.0, 9.0]
    2     3      3      8       [4.0, 5.0]      [9.0, 10.0]
    3     4      4      9       [nan, nan]       [nan, nan]
    4     5      5     10       [nan, nan]       [nan, nan]
    

Custom Functions
~~~~~~~~~~~~~~~~

To use the transform function, you have to create your custom functions.
The input data will be passed as dict of list, and the output data
should be either pandas DataFrame, pandas Series, numpy ndArray or list.
Note, the output length should be in consist with the orignal data
length.

For exmaple, this function takes input dictionary data and sum them up.
The final output is a list.

.. code:: ipython3

    import copy
    def list_output (dataDict):
        res = []
        for i in dataDict:
            if len(res) == 0:
                res = copy.deepcopy(dataDict[i])
                continue
            for ix,v in enumerate(dataDict[i]):
                res[ix] += v
        return res

.. code:: ipython3

    trans = tst.Time_Series_Transformer(data,timeSeriesCol='time')
    trans = trans.transform(
        inputLabels = ['data1','data2'],
        newName = 'sumCol',
        func = list_output
    )
    print(trans.to_pandas())


.. parsed-literal::

       time  data1  data2  sumCol
    0     1      1      6       7
    1     2      2      7       9
    2     3      3      8      11
    3     4      4      9      13
    4     5      5     10      15
    

The following example will output as pandas DataFrame and also takes
additional parameters. Note: since pandas already has column name, the
new name will automatically beocme suffix.

.. code:: ipython3

    def pandas_output(dataDict, pandasColName):
        res = []
        for i in dataDict:
            if len(res) == 0:
                res = copy.deepcopy(dataDict[i])
                continue
            for ix,v in enumerate(dataDict[i]):
                res[ix] += v
        return pd.DataFrame({pandasColName:res})

.. code:: ipython3

    trans = tst.Time_Series_Transformer(data,timeSeriesCol='time')
    trans = trans.transform(
        inputLabels = ['data1','data2'],
        newName = 'sumCol',
        func = pandas_output,
        pandasColName = "pandasName"
    )
    print(trans.to_pandas())


.. parsed-literal::

       time  data1  data2  sumCol_pandasName
    0     1      1      6                  7
    1     2      2      7                  9
    2     3      3      8                 11
    3     4      4      9                 13
    4     5      5     10                 15
    

Data with Category
~~~~~~~~~~~~~~~~~~

Since time series data could be associated with different category,
Time_Series_Transformer can specify the mainCategoryCol parameter to
point out the main category. This class only provide one columns for
main category because multiple dimensions can be aggregated into a new
column as main category.

The following example has one category with two type a and b. Each of
them has some overlaped and different timestamp.

.. code:: ipython3

    data = {
        "time":[1,2,3,4,5,1,3,4,5],
        'data':[1,2,3,4,5,1,2,3,4],
        "category":['a','a','a','a','a','b','b','b','b']
    }

.. code:: ipython3

    trans = tst.Time_Series_Transformer(data,'time','category')
    trans




.. parsed-literal::

    data column
    -----------
    time
    data
    time length: 5
    category: a
    
    data column
    -----------
    time
    data
    time length: 4
    category: b
    
    main category column: category



Since we specify the main category column, data manipulation functions
can use n_jobs to execute the function in parallel. The parallel
execution is with joblib implmentation
(https://joblib.readthedocs.io/en/latest/).

.. code:: ipython3

    trans = trans.make_lag(
        inputLabels = ['data'],
        lagNum = 1,
        suffix = '_lag_',
        fillMissing = np.nan,
        n_jobs = 2,
        verbose = 10        
    )
    print(trans.to_pandas())


.. parsed-literal::

    [Parallel(n_jobs=2)]: Using backend LokyBackend with 2 concurrent workers.
    

.. parsed-literal::

       time  data  data_lag_1 category
    0     1     1         NaN        a
    1     2     2         1.0        a
    2     3     3         2.0        a
    3     4     4         3.0        a
    4     5     5         4.0        a
    5     1     1         NaN        b
    6     3     2         1.0        b
    7     4     3         2.0        b
    8     5     4         3.0        b
    

.. parsed-literal::

    [Parallel(n_jobs=2)]: Done   2 out of   2 | elapsed:    3.6s remaining:    0.0s
    [Parallel(n_jobs=2)]: Done   2 out of   2 | elapsed:    3.6s finished
    

To further support the category, there are two functions to deal with
different time length data: pad_different_category_time and
remove_different_category_time. The first function is padding the
different length into same length, while the other is remove different
timestamp.

.. code:: ipython3

    trans = tst.Time_Series_Transformer(data,'time','category')
    trans = trans.pad_different_category_time(fillMissing = np.nan
    )
    print(trans.to_pandas())


.. parsed-literal::

       time  data category
    0     1   1.0        a
    1     2   2.0        a
    2     3   3.0        a
    3     4   4.0        a
    4     5   5.0        a
    5     1   1.0        b
    6     2   NaN        b
    7     3   2.0        b
    8     4   3.0        b
    9     5   4.0        b
    

.. code:: ipython3

    trans = tst.Time_Series_Transformer(data,'time','category')
    trans = trans.remove_different_category_time()
    print(trans.to_pandas())


.. parsed-literal::

       time  data category
    0     1     1        a
    1     3     3        a
    2     4     4        a
    3     5     5        a
    4     1     1        b
    5     3     2        b
    6     4     3        b
    7     5     4        b
    

IO
--

IO is a huge component for this package. The current version support
pandas DataFrame, numpy ndArray, Apache Arrow Table, Apache Feather, and
Apache Parquet. All those io can specify whether to expand category or
time for the export format. In this demo, we will show numpy and pandas.
Also, Transformer can combine make_label function and sepLabel parameter
inside of export to seperate data and label.

pandas
~~~~~~

.. code:: ipython3

    data = {
        "time":[1,2,3,4,5,1,3,4,5],
        'data':[1,2,3,4,5,1,2,3,4],
        "category":['a','a','a','a','a','b','b','b','b']
    }
    df = pd.DataFrame(data)

.. code:: ipython3

    trans = tst.Time_Series_Transformer.from_pandas(
        pandasFrame = df,
        timeSeriesCol = 'time',
        mainCategoryCol= 'category'
    )
    trans




.. parsed-literal::

    data column
    -----------
    time
    data
    time length: 5
    category: a
    
    data column
    -----------
    time
    data
    time length: 4
    category: b
    
    main category column: category



To expand the data, all category should be in consist. Besides the pad
and remove function, we can use preprocessType parameter to achive that.

.. code:: ipython3

    print(trans.to_pandas(
        expandCategory = True,
        expandTime = False,
        preprocessType = 'pad'
    ))


.. parsed-literal::

       time  data_a  data_b
    0     1       1     1.0
    1     2       2     NaN
    2     3       3     2.0
    3     4       4     3.0
    4     5       5     4.0
    

.. code:: ipython3

    print(trans.to_pandas(
        expandCategory = False,
        expandTime = True,
        preprocessType = 'pad'
    ))


.. parsed-literal::

       data_1  data_2  data_3  data_4  data_5 category
    0       1     2.0       3       4       5        a
    1       1     NaN       2       3       4        b
    

.. code:: ipython3

    print(trans.to_pandas(
        expandCategory = True,
        expandTime = True,
        preprocessType = 'pad'
    ))


.. parsed-literal::

       data_a_1  data_b_1  data_a_2  data_b_2  data_a_3  data_b_3  data_a_4  \
    0         1       1.0         2       NaN         3       2.0         4   
    
       data_b_4  data_a_5  data_b_5  
    0       3.0         5       4.0  
    

make_label function can be used with sepLabel parameter. This function
can be used for seperating X and y for machine learning cases.

.. code:: ipython3

    trans = trans.make_lead('data',leadNum = 1,suffix = '_lead_')
    trans = trans.make_label("data_lead_1")

.. code:: ipython3

    data, label = trans.to_pandas(
        expandCategory = False,
        expandTime = False,
        preprocessType = 'pad',
        sepLabel = True
    )

.. code:: ipython3

    print(data)


.. parsed-literal::

       time  data category
    0     1   1.0        a
    1     2   2.0        a
    2     3   3.0        a
    3     4   4.0        a
    4     5   5.0        a
    5     1   1.0        b
    6     2   NaN        b
    7     3   2.0        b
    8     4   3.0        b
    9     5   4.0        b
    

.. code:: ipython3

    print(label)


.. parsed-literal::

       data_lead_1
    0          2.0
    1          3.0
    2          4.0
    3          5.0
    4          NaN
    5          2.0
    6          NaN
    7          3.0
    8          4.0
    9          NaN
    

numpy
~~~~~

Since numpy has no column name, it has to use index number to specify
column.

.. code:: ipython3

    data = {
        "time":[1,2,3,4,5,1,3,4,5],
        'data':[1,2,3,4,5,1,2,3,4],
        "category":['a','a','a','a','a','b','b','b','b']
    }
    npArray = pd.DataFrame(data).values

.. code:: ipython3

    trans = tst.Time_Series_Transformer.from_numpy(
        numpyData= npArray,
        timeSeriesCol = 0,
        mainCategoryCol = 2)
    trans




.. parsed-literal::

    data column
    -----------
    0
    1
    time length: 5
    category: a
    
    data column
    -----------
    0
    1
    time length: 4
    category: b
    
    main category column: 2



.. code:: ipython3

    trans = trans.make_lead(1,leadNum = 1,suffix = '_lead_')
    trans = trans.make_label("1_lead_1")

.. code:: ipython3

    X,y = trans.to_pandas(
        expandCategory = False,
        expandTime = False,
        preprocessType = 'pad',
        sepLabel = True
    )

.. code:: ipython3

    print(X)


.. parsed-literal::

       0    1  2
    0  1  1.0  a
    1  2  2.0  a
    2  3  3.0  a
    3  4  4.0  a
    4  5  5.0  a
    5  1  1.0  b
    6  2  NaN  b
    7  3  2.0  b
    8  4  3.0  b
    9  5  4.0  b
    

.. code:: ipython3

    print(y)


.. parsed-literal::

       1_lead_1
    0       2.0
    1       3.0
    2       4.0
    3       5.0
    4       NaN
    5       2.0
    6       NaN
    7       3.0
    8       4.0
    9       NaN
    

Stock_Transformer
=================

Stock_Transformer is a subclass of Time_Series_Transformer. Hence, all
the function demonstrated in Time_Series_Transformer canbe used in
Stock_Transformer. The differences for Stock_Transformer is that it is
required to specify High, Low, Open, Close, Volume columns. Besides
these information, it has pandas-ta strategy implmentation to create
technical indicator (https://github.com/twopirllc/pandas-ta). Moreover,
the io class for Stock_Transformer support yfinance and investpy. We can
directly extract data from these api.

create technical indicator
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    stock = tst.Stock_Transformer.from_stock_engine_period(
        symbols = 'GOOGL',period ='1y', engine = 'yahoo'
    )
    stock




.. parsed-literal::

    data column
    -----------
    Date
    Open
    High
    Low
    Close
    Volume
    Dividends
    Stock Splits
    time length: 253
    category: None
    



.. code:: ipython3

    import pandas_ta as ta
    MyStrategy = ta.Strategy(
        name="DCSMA10",
        ta=[
            {"kind": "ohlc4"},
            {"kind": "sma", "length": 10},
            {"kind": "donchian", "lower_length": 10, "upper_length": 15},
            {"kind": "ema", "close": "OHLC4", "length": 10, "suffix": "OHLC4"},
        ]
    )

.. code:: ipython3

    stock = stock.get_technial_indicator(MyStrategy)
    print(stock.to_pandas().head())


.. parsed-literal::

             Date         Open         High          Low        Close   Volume  \
    0  2020-01-06  1351.630005  1398.319946  1351.000000  1397.810059  2338400   
    1  2020-01-07  1400.459961  1403.500000  1391.560059  1395.109985  1716500   
    2  2020-01-08  1394.819946  1411.849976  1392.630005  1405.040039  1765700   
    3  2020-01-09  1421.930054  1428.680054  1410.209961  1419.790039  1660000   
    4  2020-01-10  1429.469971  1434.939941  1419.599976  1428.959961  1312900   
    
       Dividends  Stock Splits        OHLC4  SMA_10  DCL_10_15  DCM_10_15  \
    0          0             0  1374.690002     NaN        NaN        NaN   
    1          0             0  1397.657501     NaN        NaN        NaN   
    2          0             0  1401.084991     NaN        NaN        NaN   
    3          0             0  1420.152527     NaN        NaN        NaN   
    4          0             0  1428.242462     NaN        NaN        NaN   
    
       DCU_10_15  EMA_10_OHLC4  
    0        NaN           NaN  
    1        NaN           NaN  
    2        NaN           NaN  
    3        NaN           NaN  
    4        NaN           NaN  
    

For more usage please visit our gallery
