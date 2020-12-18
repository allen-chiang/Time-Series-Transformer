import pytest
from time_series_transform.stock_transform.base import *
from time_series_transform.stock_transform.stock_extractor import *
from time_series_transform.stock_transform.util import *

###################### Data and Result ########################
@pytest.fixture('class')
def extractor_sample():
    return {
        'period': ["1y", "max", "1d"],
        'date': [["2020-07-27", "2020-08-26"]],
        'symbol': ["aapl"],
        'country': ['united states']
    }

@pytest.fixture('class')
def extractor_yahoo_expect():
    aapl = {'Date': np.array(['2020-07-27', '2020-07-28', '2020-07-29', '2020-07-30',
       '2020-07-31', '2020-08-03', '2020-08-04', '2020-08-05',
       '2020-08-06', '2020-08-07', '2020-08-10', '2020-08-11',
       '2020-08-12', '2020-08-13', '2020-08-14', '2020-08-17',
       '2020-08-18', '2020-08-19', '2020-08-20', '2020-08-21',
       '2020-08-24', '2020-08-25', '2020-08-26'], dtype=object), 'Open': np.array([ 93.38023652,  94.03542883,  93.42009733,  93.85606094,
       102.52296174, 107.81924563, 108.74847112, 108.99260671,
       110.01648845, 113.01002984, 112.40607286, 111.77715445,
       110.30719128, 114.23291727, 114.63223644, 115.86260637,
       114.15555683, 115.78274476, 115.55064397, 119.05709074,
       128.47584458, 124.48274064, 125.96268704]), 'High': np.array([ 94.57103105,  94.21728937,  94.89489261,  95.95863661,
       106.04053887, 111.24464921, 110.40013973, 110.00403435,
       114.00988593, 113.47922158, 113.57905226, 112.28876872,
       113.07991164, 115.84264311, 114.80194183, 115.88756483,
       115.8002185 , 116.96071278, 118.18859463, 124.65244058,
       128.56320302, 124.96440782, 126.77378771]), 'Low': np.array([ 93.15105014,  92.91936728,  93.38273081,  93.43754072,
       100.47020548, 107.51283258, 108.00608998, 108.51429247,
       109.41112804, 110.10254753, 109.81055224, 108.91958151,
       110.10753878, 113.73128029, 112.85030699, 113.7662247 ,
       113.81114979, 115.41088874, 115.53317228, 119.04461532,
       123.724048  , 122.84056949, 124.86707355]), 'Close': np.array([ 94.47636414,  92.92435455,  94.70555878,  95.85151672,
       105.88608551, 108.55415344, 109.27909851, 109.67519379,
       113.50167847, 110.92113495, 112.53335571, 109.18662262,
       112.81536865, 114.81192017, 114.70960236, 114.41011047,
       115.36347198, 115.50821686, 118.07129669, 124.15579987,
       125.64073944, 124.61001587, 126.30459595]), 'Volume': np.array([121214000, 103625600,  90329200, 158130000, 374336800, 308151200,
       173071600, 121992000, 202428800, 198045600, 212403600, 187902400,
       165944800, 210082000, 165565200, 119561600, 105633600, 145538000,
       126907200, 338054800, 345937600, 211495600, 163022400], dtype=int), 'Dividends': np.array([0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
       0.205, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
       0.   , 0.   , 0.   , 0.   , 0.   ]), 'Stock Splits': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0], dtype=int)}
    msft = {'Date': np.array(['2020-07-27', '2020-07-28', '2020-07-29', '2020-07-30',
       '2020-07-31', '2020-08-03', '2020-08-04', '2020-08-05',
       '2020-08-06', '2020-08-07', '2020-08-10', '2020-08-11',
       '2020-08-12', '2020-08-13', '2020-08-14', '2020-08-17',
       '2020-08-18', '2020-08-19', '2020-08-20', '2020-08-21',
       '2020-08-24', '2020-08-25', '2020-08-26'], dtype=object), 'Open': np.array([200.45934207, 202.58861208, 201.48417448, 199.9917051 ,
       203.37464204, 210.45894367, 213.09563743, 213.82197027,
       211.27481549, 213.77224513, 210.60816415, 206.12080856,
       204.26017979, 208.3893542 , 207.71276894, 208.54856189,
       209.47390145, 210.93775445, 208.99283592, 213.30156329,
       214.22912227, 212.54355925, 217.31107464]), 'High': np.array([202.94680101, 203.67314057, 203.6233831 , 203.43435506,
       204.0711427 , 216.54823869, 213.69263367, 213.92147471,
       215.28459821, 214.61797213, 210.81711738, 206.60834082,
       209.2251535 , 210.28977639, 208.53860715, 210.13058212,
       211.2947233 , 211.5461622 , 214.43858529, 215.68532185,
       214.95722702, 216.04438846, 221.51007292]), 'Low': np.array([199.85240148, 200.7279976 , 200.99662706, 198.56888583,
       198.01168103, 209.38435952, 209.2550001 , 210.50868823,
       210.48878512, 209.87189583, 205.31485961, 202.1209702 ,
       203.72289532, 207.10581693, 206.46903946, 207.87196517,
       208.16053096, 208.70359815, 208.36449135, 212.29420611,
       211.87528422, 212.54355925, 216.7924282 ]), 'Close': np.array([202.82740784, 201.0065918 , 203.03634644, 202.87715149,
       203.98158264, 215.45375061, 212.220047  , 211.87181091,
       215.26470947, 211.41412354, 207.20532227, 202.35977173,
       208.140625  , 207.65306091, 207.85206604, 209.22514343,
       210.42909241, 209.15242004, 214.01968384, 212.46376038,
       213.13200378, 215.90475464, 220.57252502]), 'Volume': np.array([30160900, 23251400, 19632600, 25079600, 51248000, 78983000,
       49280100, 28858600, 32656800, 27789600, 36716500, 36446500,
       28041400, 22588900, 17958900, 20184800, 21336200, 27627600,
       26981500, 36249300, 25460100, 23043700, 39600800], dtype=int), 'Dividends': np.array([0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
       0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.51, 0.  , 0.  , 0.  , 0.  ,
       0.  ]), 'Stock Splits': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0], dtype=int)}
    res = {'aapl': Time_Series_Data(aapl,'Date'),
            'msft':Time_Series_Data(msft,'Date')}
    return res

@pytest.fixture('class')
def extractor_investing_expect():
    aapl = {'Date': np.array(['2020-07-27', '2020-07-28', '2020-07-29', '2020-07-30',
       '2020-07-31', '2020-08-03', '2020-08-04', '2020-08-05',
       '2020-08-06', '2020-08-07', '2020-08-10', '2020-08-11',
       '2020-08-12', '2020-08-13', '2020-08-14', '2020-08-17',
       '2020-08-18', '2020-08-19', '2020-08-20', '2020-08-21',
       '2020-08-24', '2020-08-25', '2020-08-26'], dtype=object), 'Open': np.array([ 93.71,  94.37,  93.75,  94.19, 102.88, 108.2 , 109.13, 109.38,
       110.41, 113.2 , 112.6 , 111.97, 110.5 , 114.43, 114.83, 116.06,
       114.35, 115.98, 115.75, 119.26, 128.7 , 124.7 , 126.18]), 'High': np.array([ 94.91,  94.55,  95.23,  96.3 , 106.42, 111.64, 110.79, 110.39,
       114.41, 113.67, 113.78, 112.48, 113.28, 116.04, 115.  , 116.09,
       116.  , 117.16, 118.39, 124.87, 128.78, 125.18, 126.99]), 'Low': np.array([ 93.48,  93.25,  93.71,  93.77, 100.83, 107.89, 108.39, 108.9 ,
       109.8 , 110.29, 110.  , 109.11, 110.3 , 113.93, 113.05, 113.96,
       114.01, 115.61, 115.73, 119.25, 123.94, 123.05, 125.08]), 'Close': np.array([ 94.81,  93.25,  95.04,  96.19, 106.26, 108.94, 109.67, 110.06,
       113.9 , 111.11, 112.73, 109.38, 113.01, 115.01, 114.91, 114.61,
       115.56, 115.71, 118.28, 124.37, 125.86, 124.83, 126.52]), 'Volume': np.array([121214192, 103625504,  90329256, 158130016, 374295456, 308151392,
       172792368, 121991952, 202428896, 198045616, 212403424, 187902368,
       165944816, 210082064, 165565216, 119561440, 105633536, 145538016,
       126907184, 338054656, 345937760, 211495792, 163022272], dtype=int), 'Currency': np.array(['USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD',
       'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD',
       'USD', 'USD', 'USD', 'USD', 'USD'], dtype=object)}
    msft = {'Date': np.array(['2020-07-27', '2020-07-28', '2020-07-29', '2020-07-30',
       '2020-07-31', '2020-08-03', '2020-08-04', '2020-08-05',
       '2020-08-06', '2020-08-07', '2020-08-10', '2020-08-11',
       '2020-08-12', '2020-08-13', '2020-08-14', '2020-08-17',
       '2020-08-18', '2020-08-19', '2020-08-20', '2020-08-21',
       '2020-08-24', '2020-08-25', '2020-08-26'], dtype=object), 'Open': np.array([201.87, 203.87, 202.5 , 201.07, 204.4 , 211.69, 214.27, 214.65,
       212.45, 214.85, 211.38, 207.16, 205.37, 209.57, 208.76, 209.68,
       210.63, 211.56, 209.54, 213.86, 214.8 , 213.1 , 217.85]), 'High': np.array([203.97, 204.67, 204.65, 204.41, 205.1 , 217.62, 214.76, 214.65,
       216.37, 215.7 , 211.38, 207.62, 210.25, 211.33, 209.59, 211.18,
       212.35, 212.09, 214.99, 216.25, 215.51, 216.52, 222.08]), 'Low': np.array([200.89, 201.95, 202.01, 199.64, 199.01, 210.54, 210.36, 211.57,
       211.61, 210.93, 206.35, 203.14, 204.82, 208.21, 207.51, 208.94,
       209.22, 209.39, 208.96, 212.85, 212.49, 213.1 , 217.4 ]), 'Close': np.array([203.85, 202.02, 204.06, 203.9 , 205.01, 216.54, 213.29, 212.94,
       216.35, 212.48, 208.25, 203.38, 209.19, 208.7 , 208.9 , 210.28,
       211.49, 209.7 , 214.58, 213.02, 213.69, 216.47, 221.15]), 'Volume': np.array([30160868, 23251388, 19632602, 25079596, 51247968, 78983008,
       49280056, 28858620, 32656844, 27820420, 36716464, 36446460,
       28041364, 22588870, 17958936, 20184756, 21336168, 27627560,
       26981478, 36249320, 25460148, 23043696, 39600828], dtype=int), 'Currency': np.array(['USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD',
       'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD',
       'USD', 'USD', 'USD', 'USD', 'USD'], dtype=object)}

    res = {'aapl': Time_Series_Data(aapl,'Date'),
            'msft':Time_Series_Data(msft,'Date')}
    return res

@pytest.fixture('class')
def extractor_portfolio_sample():
    return {
        'period': ["1y", "max", "1d"],
        'date': [["2020-07-27", "2020-08-26"]],
        'symbol': ["aapl",'msft'],
        'country': ['united states', 'united states']
    }


@pytest.fixture('class')
def util_scaler_sample():
    data = [[], [100], [100,20,30], list(range(30))]
    return data

@pytest.fixture('class')
def util_stock_sample():
    data = {'Date': np.array(['2019-11-20', '2019-11-21', '2019-11-22', '2019-11-25',
        '2019-11-26', '2019-11-27', '2019-11-29', '2019-12-02',
        '2019-12-03', '2019-12-04', '2019-12-05', '2019-12-06',
        '2019-12-09', '2019-12-10', '2019-12-11', '2019-12-12',
        '2019-12-13', '2019-12-16', '2019-12-17', '2019-12-18'],
       dtype='<U10'),
            'Open': np.array([65.28211731, 64.82728897, 64.55686221, 64.58636831, 65.62630751,
                    65.29194888, 65.54271583, 65.70741992, 63.50464714, 64.18317973,
                    64.85188633, 65.75905139, 66.37858796, 66.03439846, 66.08602908,
                    65.83280459, 66.73753195, 68.09951742, 68.73134069, 68.78788252]),
            'High': np.array([65.41486909, 64.90596168, 64.70191081, 65.50337897, 65.68039408,
                    65.88198683, 65.88689963, 65.9483523 , 63.80458035, 64.73387376,
                    65.36816577, 66.62442872, 66.57526226, 66.39579327, 66.64901994,
                    67.00795108, 67.68158222, 69.03127826, 69.27219875, 69.30416193]),
            'Low': np.array([64.01846201, 64.21021162, 64.12663063, 64.53965685, 64.53474775,
                    65.22557298, 65.37062018, 64.76829159, 63.00803961, 64.08729594,
                    64.5912895 , 65.7147933 , 65.12722956, 65.36077368, 66.00981723,
                    65.71971725, 66.60723352, 68.09460319, 68.542034  , 68.62070862]),
            'Close': np.array([64.70437622, 64.41426849, 64.35772705, 65.48616791, 64.97481537,
                    65.8475647 , 65.70251465, 64.94284058, 63.78491592, 64.34789276,
                    65.29194641, 66.5531311 , 65.62138367, 66.00489807, 66.56788635,
                    66.73751831, 67.64470673, 68.80263519, 68.93785095, 68.77313232]),
            'Volume': np.array([106234400, 121395200,  65325200,  84020400, 105207600,  65235600,
                    46617600,  94487200, 114430400,  67181600,  74424400, 106075600,
                    128042400,  90420400,  78756800, 137310400, 133587600, 128186000,
                    114158400, 116028400]),
            'Dividends': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.]),
            'Stock Splits': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.])}
    data = pd.DataFrame(data)
    out = [[1,2,3], data]
    return out

@pytest.fixture('class')
def util_macd_output():
    out = {True:[[],[np.array(0)],
    np.array([ 0.        , -0.7977208 , -0.57822641]),
    np.array([0.        , 0.00997151, 0.0278164 , 0.05247809, 0.0828554 ,
       0.11783225, 0.1563057 , 0.19721132, 0.23954485, 0.2823797 ,
       0.32488004, 0.36630945, 0.40603556, 0.44353089, 0.47837069,
       0.51022811, 0.53886758, 0.56413675, 0.58595762, 0.60431723,
       0.61925843, 0.6308707 , 0.63928167, 0.64464905, 0.64715343,
       0.64699177, 0.64437167, 0.63950645, 0.63261098, 0.62389814])],
           False:[[], {'EMA_12': np.array([100.]),
                    'EMA_26': np.array([100.]),
                    'DIF': np.array([0.]),
                    'DEM': np.array([0.]),
                    'OSC': np.array([0.])},
                    {'EMA_12': np.array([100.        ,  56.66666667,  46.25866051]),
                    'EMA_26': np.array([100.        ,  58.46153846,  48.23558403]),
                    'DIF': np.array([ 0.        , -1.79487179, -1.97692352]),
                    'DEM': np.array([ 0.        , -0.997151  , -1.39869711]),
                    'OSC': np.array([ 0.        , -0.7977208 , -0.57822641])},
                    {'EMA_12': np.array([ 0.        ,  0.54166667,  1.1108545 ,  1.70718391,  2.33013385,
                            2.97905178,  3.65316581,  4.35159852,  5.07338219,  5.81747474,
                            6.58277601,  7.36814394,  8.17241011,  8.99439453,  9.83291914,
                            10.68681999, 11.55495793, 12.43622764, 13.32956518, 14.23395391,
                            15.14842892, 16.07208019, 17.00405443, 17.94355591, 18.88984628,
                            19.84224366, 20.80012112, 21.76290456, 22.73007028, 23.70114227]),
                    'EMA_26': np.array([ 0.        ,  0.51923077,  1.05125678,  1.59604022,  2.15352841,
                            2.72365397,  3.30633512,  3.90147606,  4.50896728,  5.12868609,
                            5.76049704,  6.40425254,  7.05979334,  7.72694924,  8.40553968,
                            9.0953744 ,  9.79625419, 10.50797158, 11.23031153, 11.96305221,
                            12.70596571, 13.4588188 , 14.22137363, 14.99338848, 15.77461845,
                            16.56481618, 17.36373247, 18.17111701, 18.98671895, 19.81028752]),
                    'DIF': np.array([0.        , 0.0224359 , 0.05959773, 0.11114369, 0.17660544,
                            0.25539782, 0.34683068, 0.45012246, 0.56441491, 0.68878865,
                            0.82227897, 0.9638914 , 1.11261677, 1.26744529, 1.42737947,
                            1.5914456 , 1.75870373, 1.92825606, 2.09925365, 2.2709017 ,
                            2.44246321, 2.61326139, 2.7826808 , 2.95016743, 3.11522782,
                            3.27742749, 3.43638865, 3.59178754, 3.74335132, 3.89085475]),
                    'DEM': np.array([0.        , 0.01246439, 0.03178133, 0.0586656 , 0.09375003,
                            0.13756557, 0.19052499, 0.25291114, 0.32487006, 0.40640895,
                            0.49739893, 0.59758195, 0.70658121, 0.82391439, 0.94900878,
                            1.08121748, 1.21983615, 1.3641193 , 1.51329604, 1.66658447,
                            1.82320478, 1.98239069, 2.14339914, 2.30551838, 2.46807439,
                            2.63043572, 2.79201698, 2.95228109, 3.11074034, 3.26695661]),
                    'OSC': np.array([0.        , 0.00997151, 0.0278164 , 0.05247809, 0.0828554 ,
                            0.11783225, 0.1563057 , 0.19721132, 0.23954485, 0.2823797 ,
                            0.32488004, 0.36630945, 0.40603556, 0.44353089, 0.47837069,
                            0.51022811, 0.53886758, 0.56413675, 0.58595762, 0.60431723,
                            0.61925843, 0.6308707 , 0.63928167, 0.64464905, 0.64715343,
                            0.64699177, 0.64437167, 0.63950645, 0.63261098, 0.62389814])}]}
    return out


@pytest.fixture('class')
def util_stochastic_oscillator_output():
    out = [[], {'k_val': np.array([        np.nan,         np.nan,         np.nan,         np.nan,         np.nan,
                np.nan,         np.nan,         np.nan,         np.nan,         np.nan,
                np.nan,         np.nan,         np.nan, 82.86880556, 97.77165536,
        93.23903111, 99.21097349, 96.20398455, 94.66252704, 91.56576726]),
 'd_val': np.array([        np.nan,         np.nan,         np.nan,         np.nan,         np.nan,
                np.nan,         np.nan,         np.nan,         np.nan,         np.nan,
                np.nan,         np.nan,         np.nan,         np.nan,         np.nan,
        91.29316401, 96.74055332, 96.21799638, 96.69249503, 94.14409295])}]
    return out

@pytest.fixture('class')
def util_rsi_output():
    out = [[],np.array([np.nan]),
        np.array([       np.nan, 0.        , 0.95238095]),
        np.array([ np.nan, 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.,
                100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.,
                100., 100., 100., 100., 100., 100., 100., 100.])]
    return out

@pytest.fixture('class')
def util_william_r_output():
    out = [[],np.array([         np.nan,          np.nan,          np.nan,          np.nan,
                np.nan,          np.nan,          np.nan,          np.nan,
                np.nan,          np.nan,          np.nan,          np.nan,
                np.nan, -17.13119444,  -2.22834464,  -6.76096889,
        -0.78902651,  -3.79601545,  -5.33747296,  -8.43423274])]
    return out

@pytest.fixture(scope = 'class')
def base_stock_test_sample():
    se = Stock_Extractor('aapl','yahoo').get_period('1y')
    return se

@pytest.fixture(scope = 'class')
def base_portfolio_test_sample():
    stockList = ["aapl", "0050.TW", "MSFT"]
    pe = Portfolio_Extractor(stockList,'yahoo').get_period('1y')
    return pe

###################### Helper Functions ########################

def compare_equal(a,b):
    try:
        np.testing.assert_equal(a,b)
    except AssertionError:
        return False
    return True

def compare_arr_result(real, expect):
    if isinstance(real, Time_Series_Data):
        indx = 0
        for _ in real:
            for key in real[indx]:
                if real[indx][key] != expect[indx][key]:
                    if real[indx][key].round(4) != expect[indx][key].round(4):
                        return False
            indx += 1
        return True

    if isinstance(real, dict):
        if list(real.keys()) != list(expect.keys()):
            return False
        for key in real:
            if not compare_equal(np.array(real[key]).round(4), np.array(expect[key]).round(4)):
                return False
        return True
    else:
        return compare_equal(np.array(real).round(4), np.array(expect).round(4))


###################### Unit Test ########################

class Test_stock_extractor:

    def test_yahoo_extractor_data(self,extractor_sample, extractor_yahoo_expect):
        date = extractor_sample['date'][0]
        symbol = extractor_sample['symbol'][0]
        se = Stock_Extractor(symbol, 'yahoo')
        data = se.get_date(date[0],date[1])
        assert compare_arr_result(data, extractor_yahoo_expect[symbol])
    
    def test_investing_extractor_data(self,extractor_sample, extractor_investing_expect):
        date = extractor_sample['date'][0]
        symbol = extractor_sample['symbol'][0]
        country = extractor_sample['country'][0]
        se = Stock_Extractor(symbol, 'investing', country = country)
        data = se.get_date(date[0],date[1])
        assert data == extractor_investing_expect[symbol]

    def test_stock_extractor_period(self,extractor_sample):
        period = extractor_sample['period']
        symbol = extractor_sample['symbol']
        country = extractor_sample['country']
        sources = ['yahoo','investing']
        for source in sources:
            for i in range(len(period)):
                for j in range(len(symbol)):
                    if source == 'investing':
                        se = Stock_Extractor(symbol[j],source,country = country[j])
                    else:
                        se = Stock_Extractor(symbol[j],source)
                    data = se.get_period(period[i])
                    assert isinstance(data, Stock)

class Test_portfolio_extractor:

    def test_yahoo_portfolio_extractor_data(self,extractor_portfolio_sample,extractor_yahoo_expect):
        date = extractor_portfolio_sample['date'][0]
        symbol = extractor_portfolio_sample['symbol']
        pe = Portfolio_Extractor(symbol, 'yahoo')
        data = pe.get_date(date[0],date[1])
        for sym in data:
            assert compare_arr_result(data[sym], extractor_yahoo_expect[sym])

    def test_investing_portfolio_extractor_data(self,extractor_portfolio_sample,extractor_investing_expect):
        date = extractor_portfolio_sample['date'][0]
        symbol = extractor_portfolio_sample['symbol']
        country = extractor_portfolio_sample['country']
        pe = Portfolio_Extractor(symbol, 'yahoo', country = country)
        data = pe.get_date(date[0],date[1])
        for sym in data:
            assert data[sym] == extractor_investing_expect[sym]

    def test_yahoo_portfolio_period(self,extractor_portfolio_sample):
        period = extractor_portfolio_sample['period']
        stockList = extractor_portfolio_sample['symbol']
        source = 'yahoo'
        for i in range(len(period)):
            pe = Portfolio_Extractor(stockList,source)
            data = pe.get_period(period[i])
            assert isinstance(data, Portfolio)


    def test_investing_portfolio_period(self,extractor_portfolio_sample):
        period = extractor_portfolio_sample['period']
        country = extractor_portfolio_sample['country']
        stockList = extractor_portfolio_sample['symbol']
        source = 'investing'
        for i in range(len(stockList)):
            pe = Portfolio_Extractor(stockList,source, country = country)
            data = pe.get_period(period[i])
            assert isinstance(data, Portfolio)



class Test_stock_util:
    
    def test_macd(self, util_scaler_sample, util_macd_output):
        return_diff = [True, False]
        for rd in return_diff:
            for ind in range(len(util_scaler_sample)):
                ar = util_scaler_sample[ind]
                out_res = util_macd_output[rd][ind]
                macd_res = []
                oriLen = len(ar)
                if oriLen == 0:
                    with pytest.raises(ValueError):
                        macd_res = macd(ar, rd)
                else:
                    macd_res = macd(ar, rd)
                    assert compare_arr_result(macd_res, out_res)
                    


    def test_stochastic_oscillator(self, util_stock_sample, util_stochastic_oscillator_output):
        for ind in range(len(util_stock_sample)):
            ar = util_stock_sample[ind]
            so_out = util_stochastic_oscillator_output[ind]
            so_res = []
            
            if not isinstance(ar, Time_Series_Data) and not isinstance(ar, pd.DataFrame):
                with pytest.raises(ValueError):
                    so_res = stochastic_oscillator(ar)
            else:
                so_res = stochastic_oscillator(ar)
                assert compare_arr_result(so_res, so_out)

    def test_rsi(self, util_scaler_sample, util_rsi_output):
        for ind in range(len(util_scaler_sample)):
            ar = util_scaler_sample[ind]
            rsi_out = util_rsi_output[ind]
            rsi_res = []
            oriLen = len(ar)
            if oriLen == 0:
                with pytest.raises(ValueError):
                    rsi_res = rsi(ar)
            else:
                rsi_res = rsi(ar)
                assert compare_arr_result(rsi_res, rsi_out)

    def test_williams_r(self, util_stock_sample, util_william_r_output):
        for ind in range(len(util_stock_sample)):
            ar = util_stock_sample[ind]
            w_out = util_william_r_output[ind]
            w_res = []

            if not isinstance(ar, Time_Series_Data) and not isinstance(ar, pd.DataFrame):
                with pytest.raises(ValueError):
                    w_res = williams_r(ar)
            else:
                w_res = williams_r(ar)
                assert compare_arr_result(w_res, w_out)
