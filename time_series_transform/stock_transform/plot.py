import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from time_series_transform.stock_transform.base import *

class Plot(object):
    def __init__(self, f, stock):
        print("init")
        self.f = f

    def __call__(self):
        print("hihi")
        self.f()

@Plot
def aFunc():
    print("afunc")