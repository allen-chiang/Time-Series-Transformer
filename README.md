# Time Series Transformer


[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
![Build](https://github.com/allen-chiang/Time-Series-Transformer/workflows/Build/badge.svg)
[![Build Status](https://dev.azure.com/kuanlunchiang/Time%20Series%20Transformer/_apis/build/status/allen-chiang.Time-Series-Transformer?branchName=master)](https://dev.azure.com/kuanlunchiang/Time%20Series%20Transformer/_build/latest?definitionId=3&branchName=master)
[![Board Status](https://dev.azure.com/kuanlunchiang/4514fff7-ad24-4603-9373-c28efeaada71/b19741c8-3782-44ee-8a92-2805fbeb49f9/_apis/work/boardbadge/e0f238c1-381a-4686-a599-43174bf8237f)](https://dev.azure.com/kuanlunchiang/4514fff7-ad24-4603-9373-c28efeaada71/_boards/board/t/b19741c8-3782-44ee-8a92-2805fbeb49f9/Microsoft.RequirementCategory)

Time Series Transformer is designed to handle time series data pre-processing for machine learning and deep learning. The general use case includes making lag/lead features, denoising data, and making various moving average (geometric and arithmetic ). Furthermore, this package provides extra features to different time series data area, including:

1. stock
    - extract data from different engine (only support yahoo finance in the current version)
    - plot interative charts for stock analsys
    - various technical indicator transformation i.e. MACD, stochastic oscillator, William %, relative strength index
    
## Pre-Processing for Machine Learning



```python
import numpy as np
import pandas as pd
import time_series_transform as tst
from time_series_transform.transform_core_api.time_series_transformer import Pandas_Time_Series_Panel_Dataset
```

This example uses the stock api to extract stock googl for one year. Subsequently, it shows how to create multiple lags data as predictors and lead data as target variable.


```python
stock_ext = tst.Stock_Extractor('googl','yahoo')
df = stock_ext.get_stock_period(period='1y').df
print(df.head())
```

             Date     Open     High      Low    Close   Volume  Dividends  \
    0  2019-08-19  1191.83  1209.39  1190.40  1200.44  1222500          0   
    1  2019-08-20  1195.35  1198.00  1183.05  1183.53  1010300          0   
    2  2019-08-21  1195.82  1200.56  1187.92  1191.58   707600          0   
    3  2019-08-22  1193.80  1198.78  1178.91  1191.52   867600          0   
    4  2019-08-23  1185.17  1195.67  1150.00  1153.58  1812700          0   
    
       Stock Splits  
    0             0  
    1             0  
    2             0  
    3             0  
    4             0  
    

Pandas_Time_Series_Panel_Dataset takes dataframe as input. make_slide_window is the function used for making multiple lagging data, and make_lead_column is to create lead data for specific column. indexCol is the time series column, and this column is used for sorting the dataframe before lag/lead feature generations.


```python
panel_transform = Pandas_Time_Series_Panel_Dataset(df)
panel_transform.make_slide_window(
    indexCol= 'Date',windowSize = 2,colList = ['Open','High','Low','Close']
).make_lead_column(indexCol = 'Date',baseCol = 'Open',leadNum=1)
panel_transform.df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 252 entries, 251 to 0
    Data columns (total 17 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   Date          252 non-null    object 
     1   Open          252 non-null    float64
     2   High          252 non-null    float64
     3   Low           252 non-null    float64
     4   Close         252 non-null    float64
     5   Volume        252 non-null    int64  
     6   Dividends     252 non-null    int64  
     7   Stock Splits  252 non-null    int64  
     8   Open_lag1     251 non-null    float64
     9   Open_lag2     250 non-null    float64
     10  High_lag1     251 non-null    float64
     11  High_lag2     250 non-null    float64
     12  Low_lag1      251 non-null    float64
     13  Low_lag2      250 non-null    float64
     14  Close_lag1    251 non-null    float64
     15  Close_lag2    250 non-null    float64
     16  Open_lead1    251 non-null    float64
    dtypes: float64(13), int64(3), object(1)
    memory usage: 35.4+ KB
    

To obtain the data, the df attribute of Pandas_Time_Series_Panel_Dataset can be retrieved.


```python
lead_lag_stock = panel_transform.df
print(lead_lag_stock[['Date','Open','Open_lag1','Open_lead1']].sort_values('Date').head())
```

             Date     Open  Open_lag1  Open_lead1
    0  2019-08-19  1191.83        NaN     1195.35
    1  2019-08-20  1195.35    1191.83     1195.82
    2  2019-08-21  1195.82    1195.35     1193.80
    3  2019-08-22  1193.80    1195.82     1185.17
    4  2019-08-23  1185.17    1193.80     1159.45
    

Sometimes, there cuold be different categories or item in the dataset. Pandas_Time_Series_Panel_Dataset the groupby parameter can serve the advanced data manipulation for lead and lag data making. The following example is going to construct a dataframe with multiple stocks, and each stock can be represented as one item.


```python
df = tst.Portfolio_Extractor(['googl','aapl'],'yahoo').get_portfolio_period('1y').get_portfolio_dataFrame()
print(df.head())
```

             Date     Open     High      Low    Close   Volume  Dividends  \
    0  2019-08-19  1191.83  1209.39  1190.40  1200.44  1222500        0.0   
    1  2019-08-20  1195.35  1198.00  1183.05  1183.53  1010300        0.0   
    2  2019-08-21  1195.82  1200.56  1187.92  1191.58   707600        0.0   
    3  2019-08-22  1193.80  1198.78  1178.91  1191.52   867600        0.0   
    4  2019-08-23  1185.17  1195.67  1150.00  1153.58  1812700        0.0   
    
       Stock Splits symbol  
    0             0  googl  
    1             0  googl  
    2             0  googl  
    3             0  googl  
    4             0  googl  
    


```python
panel_transform = Pandas_Time_Series_Panel_Dataset(df)
panel_transform.make_slide_window(
    indexCol= 'Date',windowSize = 2,colList = ['Open','High','Low','Close'],groupby='symbol'
).make_lead_column(indexCol = 'Date',baseCol = 'Open',leadNum=1,groupby='symbol')
panel_transform.df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 504 entries, 251 to 0
    Data columns (total 18 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   Date          504 non-null    object 
     1   Open          504 non-null    float64
     2   High          504 non-null    float64
     3   Low           504 non-null    float64
     4   Close         504 non-null    float64
     5   Volume        504 non-null    int64  
     6   Dividends     504 non-null    float64
     7   Stock Splits  504 non-null    int64  
     8   symbol        504 non-null    object 
     9   Open_lag1     502 non-null    float64
     10  Open_lag2     500 non-null    float64
     11  High_lag1     502 non-null    float64
     12  High_lag2     500 non-null    float64
     13  Low_lag1      502 non-null    float64
     14  Low_lag2      500 non-null    float64
     15  Close_lag1    502 non-null    float64
     16  Close_lag2    500 non-null    float64
     17  Open_lead1    502 non-null    float64
    dtypes: float64(14), int64(2), object(2)
    memory usage: 74.8+ KB
    


```python
lead_lag_stock = panel_transform.df
print(lead_lag_stock[['Date','symbol','Open','Open_lag1','Open_lead1']].sort_values('Date').head())
```

             Date symbol     Open  Open_lag1  Open_lead1
    0  2019-08-19   aapl   208.55        NaN      208.81
    0  2019-08-19  googl  1191.83        NaN     1195.35
    1  2019-08-20   aapl   208.81     208.55      210.90
    1  2019-08-20  googl  1195.35    1191.83     1195.82
    2  2019-08-21  googl  1195.82    1195.35     1193.80
    

Note: Some other use cases could be inventory. Inventory data is usually associate with multiple categories such as item name or locations. To use groupby parameter, it has to be combined into on column, for example, item, location --> item_location. The currently api only supports one column groupby.

## Deep Learning

Transforming panel data into tensor data for deep learning model might wirte server lines of code. Using Pandas_Time_Series_Tensor_Dataset can easily complete those tidious tasks. This class will take your pandas frame as input and following the configuration to manipulate the data and make the generator for training.

The configuration can be simply setup by set_config function. There are three type of manipulation sequence --> making lagging data, category --> making a sequence of same data, and label --> making 1 step lead data. The following example uses a simple dataframe for demonstration.


```python
from time_series_transform.transform_core_api.time_series_transformer import Pandas_Time_Series_Tensor_Dataset
df = pd.DataFrame({'time':[1,2,3,4],'demand':[1,2,3,4],'category':[1,1,2,2]})
print(df)
```

       time  demand  category
    0     1       1         1
    1     2       2         1
    2     3       3         2
    3     4       4         2
    

To make the generator, there are two steps:
1. expand data from time, demand, category to category_demand_time (use expand_dataFrame_by_date to achieve this step)
2. setup configuration


```python
tensor_generator = Pandas_Time_Series_Tensor_Dataset(df)
tensor_generator.expand_dataFrame_by_date(
    categoryCol = 'category',timeSeriesCol = 'time',byCategory=False
)
print(tensor_generator.df)
```

       1_demand_1  1_demand_2  2_demand_3  2_demand_4
    0           1           2           3           4
    


```python
tensor_generator.set_config(
    name = 'demand_lag',
    colNames = ["1_demand_1"  ,"1_demand_2" , "2_demand_3" , "2_demand_4"],
    tensorType= 'sequence',
    windowSize = 2,
    sequence_stack=None, 
    isResponseVar=False, 
    seqSize=4,
    outType=np.float32
)
tensor_generator.set_config(
    name = 'demand_lead',
    colNames = ["1_demand_1"  ,"1_demand_2" , "2_demand_3" , "2_demand_4"],
    tensorType= 'label',
    windowSize = 2,
    sequence_stack=None, 
    isResponseVar=True, 
    seqSize=4,
    outType=np.float32
)
```


```python
gen = tensor_generator.make_data_generator()
for i in gen:
    print(i)
```

    ({'demand_lag': array([[[1],
            [2]],
    
           [[2],
            [3]]])}, array([3, 4]))
    

Note: More Advance manipulation including stacking different sequence and multi-steps prediction can refer gallery.
