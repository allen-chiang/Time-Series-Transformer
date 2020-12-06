import pytest
import numpy as np
import pandas as pd
import pandas_ta as ta
from time_series_transform.stock_transform.base import (Stock,Portfolio)

@pytest.fixture('class')
def dictList_stock():
    return {
        'Date': ['2020-01-01', '2020-01-02'],
        'Close': [1, 2],
        'Open': [1, 2],
        'Low': [1, 2],
        'High': [1, 2],
        'Volume': [1, 2],
        'symbol':['AT','AT']
    }

@pytest.fixture('class')
def dictList_portfolio():
    return {
        'Date': ['2020-01-01', '2020-01-02','2020-01-01', '2020-01-02'],
        'Close': [1,2,1,2],
        'Open': [1,2,1,2],
        'Low': [1,2,1,2],
        'High': [1,2,1,2],
        'Volume': [1,2,1,2],
        'symbol':['AT','AT','GOOGL','GOOGL']
    }

class Test_Stock:
    
    def test_from_time_series_tensor(self,dictList_stock):
        pass

    def test_get_technical_indicator(self):
        pass


class Test_Portfolio:

    def test_from_time_series_collection(self,dictList_portfolio):
        pass

    def test_get_technical_indicator(self):
        pass