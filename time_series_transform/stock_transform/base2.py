import scipy
import numpy as np
import pandas as pd
from numpy.fft import *
import matplotlib.pyplot as plt
from collections import ChainMap
from joblib import Parallel, delayed
import plotly.graph_objects as go
from time_series_transform.transform_core_api.util import *
from time_series_transform.transform_core_api.base import *
from time_series_transform.io import *

class Stock()