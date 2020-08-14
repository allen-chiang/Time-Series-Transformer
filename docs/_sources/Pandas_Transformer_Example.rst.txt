Pandas Transformer
==================

Making Lags and Lead Features for Machine Learning
--------------------------------------------------

Lag data
~~~~~~~~

To train a supervised machine learning model, we can use lag data as
part of feature engineering. A lag data is its piror time step’s value.
In this package, we can use Pandas_Time_Series_Panel_Dataset to make lag
data with multiple piror time steps (window size). This example will
first create a simple pandas frame and demonstrate how to create lag
data.

.. code:: ipython3

    import numpy as np
    import pandas as pd
    import time_series_transform as tst

To create its lag data, we can use make_slide_window funciton. It has
four parameters 1. indexCol: time sereis column name, this parameter
will be used for sorting the data frame before any manipulation 2.
windowSize: how many lags will be created 3. colList: the target column
names. if None is passing, all column except groupby column and index
column will be trasformed 4. groupby: category of the data

This is a simple data frame with no category column

.. code:: ipython3

    data = pd.DataFrame({'time':[1,2,3,4,5],'x1':[1,2,3,4,5],'x2':[6,7,8,9,10]})
    print(data)
    


.. parsed-literal::

       time  x1  x2
    0     1   1   6
    1     2   2   7
    2     3   3   8
    3     4   4   9
    4     5   5  10
    

.. code:: ipython3

    time_panel = tst.Pandas_Time_Series_Panel_Dataset(data)
    print(time_panel.make_slide_window(
        indexCol = 'time',
        windowSize= 2,
        colList= ['x1']
    )) 
    # if colList is passed, only list item will be transformed


.. parsed-literal::

       time  x1  x2  x1_lag1  x1_lag2
    0     1   1   6      NaN      NaN
    1     2   2   7      1.0      NaN
    2     3   3   8      2.0      1.0
    3     4   4   9      3.0      2.0
    4     5   5  10      4.0      3.0
    

.. code:: ipython3

    time_panel = tst.Pandas_Time_Series_Panel_Dataset(data)
    print(time_panel.make_slide_window(
        indexCol = 'time',
        windowSize= 2,
        colList= None
    )) 
    # if None is passed, all column will be transformed


.. parsed-literal::

       time  x1  x2  x1_lag1  x1_lag2  x2_lag1  x2_lag2  x1_lag1_lag1  \
    0     1   1   6      NaN      NaN      NaN      NaN           NaN   
    1     2   2   7      1.0      NaN      6.0      NaN           NaN   
    2     3   3   8      2.0      1.0      7.0      6.0           1.0   
    3     4   4   9      3.0      2.0      8.0      7.0           2.0   
    4     5   5  10      4.0      3.0      9.0      8.0           3.0   
    
       x1_lag1_lag2  x1_lag2_lag1  x1_lag2_lag2  
    0           NaN           NaN           NaN  
    1           NaN           NaN           NaN  
    2           NaN           NaN           NaN  
    3           1.0           1.0           NaN  
    4           2.0           2.0           1.0  
    

In some cases, this item could associated with different categories. To
manipulate the data, we can either expand the window using
expand_dataFrame_by_category to create multiple columns before making
lag data or use groupby parameter inside of make_slide_window to produce
lag data associate with its category.

Note: expand_dataFrame_by_category and groupby parameter only support on
category column. If you have multiple category column, you can concate
each category into one new category before using this api.

.. code:: ipython3

    data = pd.DataFrame(
        {'time':[1,2,3,4,5,1,2,3,4,5],
         'x1':[1,2,3,4,5,1,2,3,4,5],
         'x2':[6,7,8,9,10,6,7,8,9,10],
         'category':[1,1,1,1,1,2,2,2,2,2]
        })
    print(data)


.. parsed-literal::

       time  x1  x2  category
    0     1   1   6         1
    1     2   2   7         1
    2     3   3   8         1
    3     4   4   9         1
    4     5   5  10         1
    5     1   1   6         2
    6     2   2   7         2
    7     3   3   8         2
    8     4   4   9         2
    9     5   5  10         2
    

expand_dataFrame_by_category function will produce columns with
feature_category columns.

.. code:: ipython3

    time_panel = tst.Pandas_Time_Series_Panel_Dataset(data)
    print(time_panel.expand_dataFrame_by_category('time','category'))
    


.. parsed-literal::

       time  x1_1  x2_1  x1_2  x2_2
    0     1     1     6     1     6
    1     2     2     7     2     7
    2     3     3     8     3     8
    3     4     4     9     4     9
    4     5     5    10     5    10
    

After expanding the column, we can safely make its lag data.

.. code:: ipython3

    print(time_panel.make_slide_window(
        indexCol = 'time',
        windowSize= 2,
        colList= None
    )) 


.. parsed-literal::

       time  x1_1  x2_1  x1_2  x2_2  x1_1_lag1  x1_1_lag2  x2_1_lag1  x2_1_lag2  \
    0     1     1     6     1     6        NaN        NaN        NaN        NaN   
    1     2     2     7     2     7        1.0        NaN        6.0        NaN   
    2     3     3     8     3     8        2.0        1.0        7.0        6.0   
    3     4     4     9     4     9        3.0        2.0        8.0        7.0   
    4     5     5    10     5    10        4.0        3.0        9.0        8.0   
    
       x1_2_lag1  x1_2_lag2  x2_2_lag1  x2_2_lag2  
    0        NaN        NaN        NaN        NaN  
    1        1.0        NaN        6.0        NaN  
    2        2.0        1.0        7.0        6.0  
    3        3.0        2.0        8.0        7.0  
    4        4.0        3.0        9.0        8.0  
    

Alternatively, you can use groupby parameter in make_slide_window
function

.. code:: ipython3

    time_panel = tst.Pandas_Time_Series_Panel_Dataset(data)
    print(time_panel.make_slide_window(
        indexCol = 'time',
        windowSize= 2,
        colList= None,
        groupby = 'category'
    ))


.. parsed-literal::

       time  x1  x2  category  x1_lag1  x1_lag2  x2_lag1  x2_lag2
    0     1   1   6         1      NaN      NaN      NaN      NaN
    5     1   1   6         2      NaN      NaN      NaN      NaN
    1     2   2   7         1      1.0      NaN      6.0      NaN
    6     2   2   7         2      1.0      NaN      6.0      NaN
    2     3   3   8         1      2.0      1.0      7.0      6.0
    7     3   3   8         2      2.0      1.0      7.0      6.0
    3     4   4   9         1      3.0      2.0      8.0      7.0
    8     4   4   9         2      3.0      2.0      8.0      7.0
    4     5   5  10         1      4.0      3.0      9.0      8.0
    9     5   5  10         2      4.0      3.0      9.0      8.0
    

Lead data
~~~~~~~~~

Like making lag data, this class provides function to create lead data.
Lead data is the the future time step value of the feature. In
supervised learning, the target variable is usually associate with its
lead value.

To make lead data we can use make_lead_column function. In this
function, there are four parameter 1. indexCol: time sereis column name,
this parameter will be used for sorting the data frame before any
manipulation 2. baseCol: the target column for transformation 3.
leadNum: lead step number 4. groupby: dealing with category column

Note: Lead function will only create on lead column.

.. code:: ipython3

    data = pd.DataFrame({'time':[1,2,3,4,5],'x1':[1,2,3,4,5],'x2':[6,7,8,9,10]})
    print(data)


.. parsed-literal::

       time  x1  x2
    0     1   1   6
    1     2   2   7
    2     3   3   8
    3     4   4   9
    4     5   5  10
    

simple case

.. code:: ipython3

    time_panel = tst.Pandas_Time_Series_Panel_Dataset(data)
    time_panel.make_lead_column('time','x1',1)




.. parsed-literal::

       time  x1  x2  x1_lead1
    4     5   5  10       NaN
    3     4   4   9       5.0
    2     3   3   8       4.0
    1     2   2   7       3.0
    0     1   1   6       2.0



category case

.. code:: ipython3

    data = pd.DataFrame(
        {'time':[1,2,3,4,5,1,2,3,4,5],
         'x1':[1,2,3,4,5,1,2,3,4,5],
         'x2':[6,7,8,9,10,6,7,8,9,10],
         'category':[1,1,1,1,1,2,2,2,2,2]
        })
    print(data)


.. parsed-literal::

       time  x1  x2  category
    0     1   1   6         1
    1     2   2   7         1
    2     3   3   8         1
    3     4   4   9         1
    4     5   5  10         1
    5     1   1   6         2
    6     2   2   7         2
    7     3   3   8         2
    8     4   4   9         2
    9     5   5  10         2
    

.. code:: ipython3

    time_panel = tst.Pandas_Time_Series_Panel_Dataset(data)
    time_panel.make_lead_column('time','x1',1,'category')




.. parsed-literal::

       time  x1  x2  category  x1_lead1
    4     5   5  10         1       NaN
    9     5   5  10         2       NaN
    3     4   4   9         1       5.0
    8     4   4   9         2       5.0
    2     3   3   8         1       4.0
    7     3   3   8         2       4.0
    1     2   2   7         1       3.0
    6     2   2   7         2       3.0
    0     1   1   6         1       2.0
    5     1   1   6         2       2.0



Making Tensor Data for Deep Learning
------------------------------------

For deep learning model, especially for cnn or rnn, tensor can be very
useful. Using Pandas_Time_Series_Tensor_Dataset, we can easily create
different types of tensor.

1. sequence: make lag data given a window size
2. label: make lead data with one step forward
3. category: make category variable corrsponding to its sequence with
   given window size
4. same: return same data list

Expand DataFrame
~~~~~~~~~~~~~~~~

To use this api, it is necessary to expand the data by its date. To
achieve, you can simply use expand_dataFrame_by_date function.

Note, if newIX is True, all time series will be label from 1 through
time series number. And, ixDict attribute will be saved in the class
object. This attribute can be used to trace the naming before and after
new index.

If byCategory is True, the row will still group by category. The columns
are only expand by time data.

.. code:: ipython3

    data = pd.DataFrame(
        {'time':['a','b','c','d','e','a','b','c','d','e'],
         'x1':[1,2,3,4,5,1,2,3,4,5],
         'x2':[6,7,8,9,10,6,7,8,9,10],
         'category':[1,1,1,1,1,2,2,2,2,2]
        })
    print(data)


.. parsed-literal::

      time  x1  x2  category
    0    a   1   6         1
    1    b   2   7         1
    2    c   3   8         1
    3    d   4   9         1
    4    e   5  10         1
    5    a   1   6         2
    6    b   2   7         2
    7    c   3   8         2
    8    d   4   9         2
    9    e   5  10         2
    

Likewise, if it is False, it will be fully expanded.

.. code:: ipython3

    tensor_transform = tst.Pandas_Time_Series_Tensor_Dataset(data)
    tensor_transform.expand_dataFrame_by_date('category','time',newIX = True,byCategory=True)
    print(tensor_transform.ixDict)
    print(tensor_transform.df)


.. parsed-literal::

    {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
              x1_1  x1_2  x1_3  x1_4  x1_5  x2_1  x2_2  x2_3  x2_4  x2_5
    category                                                            
    1            1     2     3     4     5     6     7     8     9    10
    2            1     2     3     4     5     6     7     8     9    10
    

.. code:: ipython3

    tensor_transform = tst.Pandas_Time_Series_Tensor_Dataset(data)
    ixDict = tensor_transform.expand_dataFrame_by_date('category','time',byCategory=False)
    print(tensor_transform.ixDict)
    print(tensor_transform.df)


.. parsed-literal::

    {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
       1_x1_1  1_x1_2  1_x1_3  1_x1_4  1_x1_5  1_x2_1  1_x2_2  1_x2_3  1_x2_4  \
    0       1       2       3       4       5       6       7       8       9   
    
       1_x2_5  2_x1_1  2_x1_2  2_x1_3  2_x1_4  2_x1_5  2_x2_1  2_x2_2  2_x2_3  \
    0      10       1       2       3       4       5       6       7       8   
    
       2_x2_4  2_x2_5  
    0       9      10  
    

After expand the data frame, we have to setup up its configuration. This
can be done by using set_config function. Its parameter include 1. name:
output variable name 2. colNames: column list for transformation 3.
tensorType: transformation type {‘sequence’,‘category’,‘label’} 4.
sequence_stack: whether to stack the output with other transformation 5.
isResponseVar: whether to ouput the variable in the target variable 6.
windowSize: grouping size 7. seqSize: this parameter is only used for
category transformation. It should be the total number of time sequence
8. outType: output data type, and it should be numpy data type

Lag Data
~~~~~~~~

Here we use x1 variable associate with its category and time to make lag
features. To get the transformation, we can use make_data_generator
function to get an generator object. The generator will generate the
manipulation of each row of data.

.. code:: ipython3

    tensor_transform = tst.Pandas_Time_Series_Tensor_Dataset(data)
    ixDict = tensor_transform.expand_dataFrame_by_date('category','time',byCategory=True) # demo not expand by category
    tensor_transform.set_config(
        name='x1',
        colNames=['x1_1','x1_2','x1_3','x1_4','x1_5'],
        tensorType='sequence',
        sequence_stack=None,
        isResponseVar=False,
        windowSize=2,
        seqSize=5,
        outType=np.float
    )
    gen = tensor_transform.make_data_generator()
    for i in gen:
        print(i)
        print(f"Shape:{str(i[0]['x1'].shape)}")


.. parsed-literal::

    ({'x1': array([[[1],
            [2]],
    
           [[2],
            [3]],
    
           [[3],
            [4]]])}, None)
    Shape:(3, 2, 1)
    ({'x1': array([[[1],
            [2]],
    
           [[2],
            [3]],
    
           [[3],
            [4]]])}, None)
    Shape:(3, 2, 1)
    

Lead Data
~~~~~~~~~

Since the label function will always produce one step forward list, we
can use it to create response variable.

This is one step forward data

.. code:: ipython3

    tensor_transform = tst.Pandas_Time_Series_Tensor_Dataset(data)
    ixDict = tensor_transform.expand_dataFrame_by_date('category','time',byCategory=False) # demo expanded by category
    tensor_transform.set_config(
        name='x_lead1',
        colNames=['1_x1_1','1_x1_2','1_x1_3','1_x1_4','1_x1_5'],
        tensorType='label',
        sequence_stack=None,
        isResponseVar=False,
        windowSize=2,
        seqSize=5,
        outType=np.float
    )

.. code:: ipython3

    gen = tensor_transform.make_data_generator()
    for i in gen:
        print(i)
        print(f"Shape:{str(i[0]['x_lead1'].shape)}")


.. parsed-literal::

    ({'x_lead1': array([3, 4, 5])}, None)
    Shape:(3,)
    

Categorical Data
~~~~~~~~~~~~~~~~

This is type of tensor is designed for making categorical data matching
with sequence type data.

.. code:: ipython3

    tensor_transform = tst.Pandas_Time_Series_Tensor_Dataset(data)
    ixDict = tensor_transform.expand_dataFrame_by_date('category','time',byCategory=True)
    tensor_transform.set_config(
        name='x_category',
        colNames=['x1_1'],
        tensorType='category',
        sequence_stack=None,
        isResponseVar=False,
        windowSize=2,
        seqSize=5,
        outType=np.float
    )

.. code:: ipython3

    gen = tensor_transform.make_data_generator()
    for i in gen:
        print(i)
        print(f"Shape:{str(i[0]['x_category'].shape)}")


.. parsed-literal::

    ({'x_category': array([[1],
           [1],
           [1]])}, None)
    Shape:(3, 1)
    ({'x_category': array([[1],
           [1],
           [1]])}, None)
    Shape:(3, 1)
    

Same Data
---------

To make more complicated manipulation, the data can be first
pre-processed by Pandas_Time_Series_Panel_Dataset. Subsequently, it can
be processed by Pandas_Time_Series_Tensor_Dataset. Same data will return
the same array.

.. code:: ipython3

    tensor_transform = tst.Pandas_Time_Series_Tensor_Dataset(data)
    ixDict = tensor_transform.expand_dataFrame_by_date('category','time',byCategory=True)
    tensor_transform.set_config(
        name='x_same',
        colNames=['x1_1','x1_2','x1_3','x1_4','x1_5'],
        tensorType='same',
        sequence_stack=None,
        isResponseVar=False,
        windowSize=2,
        seqSize=5,
        outType=np.float
    )
    

.. code:: ipython3

    gen = tensor_transform.make_data_generator()
    for i in gen:
        print(i)
        print(f"Shape:{str(i[0]['x_same'].shape)}")


.. parsed-literal::

    ({'x_same': array([1, 2, 3, 4, 5])}, None)
    Shape:(5,)
    ({'x_same': array([1, 2, 3, 4, 5])}, None)
    Shape:(5,)
    

Stacking Data and Target Variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes, we want to stack multiple sequence together. Using
sequence_stack parameter, we can stack data together. The shape of the
output data will be (batch size, window size, feature number).

.. code:: ipython3

    tensor_transform = tst.Pandas_Time_Series_Tensor_Dataset(data)
    ixDict = tensor_transform.expand_dataFrame_by_date('category','time',byCategory=True)
    tensor_transform.set_config(
        name='x_same',
        colNames=['x1_1','x1_2','x1_3','x1_4','x1_5'],
        tensorType='same',
        sequence_stack=None,
        isResponseVar=False,
        windowSize=2,
        seqSize=5,
        outType=np.float
    )
    tensor_transform.set_config(
        name='x_same2',
        colNames=['x1_1','x1_2','x1_3','x1_4','x1_5'],
        tensorType='same',
        sequence_stack='x_same',
        isResponseVar=False,
        windowSize=2,
        seqSize=5,
        outType=np.float
    )
    tensor_transform.set_config(
        name='x_sequence',
        colNames=['x1_1','x1_2','x1_3','x1_4','x1_5'],
        tensorType='sequence',
        sequence_stack=None,
        isResponseVar=False,
        windowSize=2,
        seqSize=5,
        outType=np.float
    )
    tensor_transform.set_config(
        name='x_sequence_2',
        colNames=['x1_1','x1_2','x1_3','x1_4','x1_5'],
        tensorType='sequence',
        sequence_stack='x_sequence',
        isResponseVar=False,
        windowSize=2,
        seqSize=5,
        outType=np.float
    )
    tensor_transform.set_config(
        name='y',
        colNames=['x1_1','x1_2','x1_3','x1_4','x1_5'],
        tensorType='label',
        sequence_stack=None,
        isResponseVar=True,
        windowSize=2,
        seqSize=5,
        outType=np.float
    )

.. code:: ipython3

    gen = tensor_transform.make_data_generator()
    tmp = next(gen)
    print(tmp)


.. parsed-literal::

    ({'x_same': array([[1, 1],
           [2, 2],
           [3, 3],
           [4, 4],
           [5, 5]]), 'x_sequence': array([[[1, 1],
            [2, 2]],
    
           [[2, 2],
            [3, 3]],
    
           [[3, 3],
            [4, 4]]])}, array([3, 4, 5]))
    

Tensorflow Adopter
==================

TFReford Writer
---------------

Time series tensor can easily getting very big and computational
expensive. Hence, time_series_transformer provides an API to create and
read TFRecord. Using TFRecord Writer will create two files. 1. TFRecord
data 2. A metadata used for TFRecord Reader (can be pickled)

To create the tfRecord file, you have to use write_tfRecord function.
While to create read MetaData you can use

.. code:: ipython3

    from time_series_transform.transform_core_api.tensorflow_adopter import TFRecord_Reader,TFRecord_Writer

.. code:: ipython3

    data = pd.DataFrame(
        {'time':['a','b','c','d','e','a','b','c','d','e'],
         'x1':[1,2,3,4,5,1,2,3,4,5],
         'x2':[6,7,8,9,10,6,7,8,9,10],
         'category':[1,1,1,1,1,2,2,2,2,2]
        })
    tensor_transform = tst.Pandas_Time_Series_Tensor_Dataset(data)
    ixDict = tensor_transform.expand_dataFrame_by_date('category','time',byCategory=True)
    tensor_transform.set_config(
        name='x_same',
        colNames=['x1_1','x1_2','x1_3','x1_4','x1_5'],
        tensorType='same',
        sequence_stack=None,
        isResponseVar=False,
        windowSize=2,
        seqSize=5,
        outType=np.float
    )
    tensor_transform.set_config(
        name='x_same2',
        colNames=['x1_1','x1_2','x1_3','x1_4','x1_5'],
        tensorType='same',
        sequence_stack='x_same',
        isResponseVar=False,
        windowSize=2,
        seqSize=5,
        outType=np.float
    )
    tensor_transform.set_config(
        name='x_sequence',
        colNames=['x1_1','x1_2','x1_3','x1_4','x1_5'],
        tensorType='sequence',
        sequence_stack=None,
        isResponseVar=False,
        windowSize=2,
        seqSize=5,
        outType=np.float
    )
    tensor_transform.set_config(
        name='x_sequence_2',
        colNames=['x1_1','x1_2','x1_3','x1_4','x1_5'],
        tensorType='sequence',
        sequence_stack='x_sequence',
        isResponseVar=False,
        windowSize=2,
        seqSize=5,
        outType=np.float
    )
    tensor_transform.set_config(
        name='y',
        colNames=['x1_1','x1_2','x1_3','x1_4','x1_5'],
        tensorType='label',
        sequence_stack=None,
        isResponseVar=True,
        windowSize=2,
        seqSize=5,
        outType=np.float
    )
    
    

.. code:: ipython3

    tr = TFRecord_Writer('./tmp.tfRecor')
    tr.write_tfRecord(tensor_transform.make_data_generator())
    metaData = tr.get_tfRecord_dtype('./meta.pickle')

TFRecord Reader
---------------

To make tfRecord into tensorflow dataset, you can use the TFReader API
with the metadata created by TFRecord_Writer. To create the dataset
object, you can use make_tfDataset function.

.. code:: ipython3

    tw = TFRecord_Reader('./tmp.tfRecor',metaData)

.. code:: ipython3

    tf_dataset = tw.make_tfDataset()

.. code:: ipython3

    for i in tf_dataset.take(1):
        print(i)


.. parsed-literal::

    {'x_same': <tf.Tensor: shape=(5, 2), dtype=float32, numpy=
    array([[1., 1.],
           [2., 2.],
           [3., 3.],
           [4., 4.],
           [5., 5.]], dtype=float32)>, 'x_sequence': <tf.Tensor: shape=(3, 2, 2), dtype=float32, numpy=
    array([[[1., 1.],
            [2., 2.]],
    
           [[2., 2.],
            [3., 3.]],
    
           [[3., 3.],
            [4., 4.]]], dtype=float32)>, 'label': <tf.Tensor: shape=(3,), dtype=float32, numpy=array([3., 4., 5.], dtype=float32)>}
    


