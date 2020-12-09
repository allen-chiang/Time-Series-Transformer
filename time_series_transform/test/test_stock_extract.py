import pytest
from time_series_transform.stock_transform.base import *
from time_series_transform.stock_transform.stock_extractor import *
from time_series_transform.stock_transform.util import *

###################### Data and Result ########################

@pytest.fixture('class')
def extractor_yahoo_sample():
    return {
        'period': ["1y", "max", "1d"],
        'date': [["2020-01-01", "2020-07-01"], ["2020-01-01", "2020-03-01"],["2020-01-01", "2020-02-01"]],
        'symbol': ["aapl", "0050.TW", "MSFT"]
    }

@pytest.fixture('class')
def extractor_investing_sample():
    return {
        'period': ["1y", "max", "1d"],
        'date': [["2020-01-01", "2020-07-01"], ["2020-01-01", "2020-03-01"],["2020-01-01", "2020-02-01"]],
        'symbol': ["aapl", "1310", "2206"],
        'country': ["united states", "taiwan", "japan"]
    }

@pytest.fixture('class')
def extractor_portfolio_yahoo_sample():
    return {
        'period': ["1y", "max", "1d"],
        'date': [["2020-01-01", "2020-07-01"], ["2020-01-01", "2020-03-01"],["2020-01-01", "2020-02-01"]],
        "stockList": [["aapl", "0050.TW", "MSFT"],[],["aapl"]]
    }

@pytest.fixture('class')
def extractor_portfolio_investing_sample():
    return {
        'period': ["1y", "max", "1d"],
        'date': [["2020-01-01", "2020-07-01"], ["2020-01-01", "2020-03-01"],["2020-01-01", "2020-02-01"]],
        "stockList": [["aapl", "1310", "2206"],[],["aapl"]],
        'country': [["united states", "taiwan", "japan"],[],["united states"]]
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
    se = Stock_Extractor('aapl','yahoo').get_stock_period('1y')
    return se

@pytest.fixture(scope = 'class')
def base_portfolio_test_sample():
    stockList = ["aapl", "0050.TW", "MSFT"]
    pe = Portfolio_Extractor(stockList,'yahoo').get_portfolio_period('1y')
    return pe

###################### Helper Functions ########################

def compare_equal(a,b):
    try:
        np.testing.assert_equal(a,b)
    except AssertionError:
        return False
    return True

def compare_arr_result(real, expect):
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

    def test_yahoo_stock_extractor_period(self,extractor_yahoo_sample):
        period = extractor_yahoo_sample['period']
        symbol = extractor_yahoo_sample['symbol']
        source = 'yahoo'
        for i in range(len(symbol)):
            se = Stock_Extractor(symbol[i],source)
            data = se.get_stock_period(period[i])
            assert isinstance(data, Stock)
            assert data.symbol == symbol[i]

    def test_yahoo_stock_extractor_date(self,extractor_yahoo_sample):
        date = extractor_yahoo_sample['date']
        symbol = extractor_yahoo_sample['symbol']
        source = 'yahoo'
        for i in range(len(symbol)):
            se = Stock_Extractor(symbol[i],source)
            data = se.get_stock_date(date[i][0], date[i][1])
            assert isinstance(data, Stock)
            assert data.symbol == symbol[i]

    def test_investing_stock_extractor_period(self,extractor_investing_sample):
        period = extractor_investing_sample['period']
        symbol = extractor_investing_sample['symbol']
        country = extractor_investing_sample['country']
        source = 'investing'
        for i in range(len(symbol)):
            se = Stock_Extractor(symbol[i],source, country = country[i])
            data = se.get_stock_period(period[i])
            assert isinstance(data, Stock)
            assert data.symbol == symbol[i]

    def test_investing_stock_extractor_date(self,extractor_investing_sample):
        date = extractor_investing_sample['date']
        symbol = extractor_investing_sample['symbol']
        country = extractor_investing_sample['country']
        source = 'investing'
        for i in range(len(symbol)):
            se = Stock_Extractor(symbol[i],source, country = country[i])
            data = se.get_stock_date(date[i][0], date[i][1])
            assert isinstance(data, Stock)
            assert data.symbol == symbol[i]

class Test_portfolio_extractor:

    def test_yahoo_portfolio_period(self,extractor_portfolio_yahoo_sample):
        period = extractor_portfolio_yahoo_sample['period']
        stockList = extractor_portfolio_yahoo_sample['stockList']
        source = 'yahoo'
        for i in range(len(stockList)):
            if len(stockList[i]) == 0:
                with pytest.raises(ValueError):
                    pe = Portfolio_Extractor(stockList[i],source)
                    data = pe.get_portfolio_period(period[i])
            else:
                pe = Portfolio_Extractor(stockList[i],source)
                data = pe.get_portfolio_period(period[i])
                assert isinstance(data, Portfolio)

    def test_yahoo_portfolio_date(self,extractor_portfolio_yahoo_sample):
        date = extractor_portfolio_yahoo_sample['date']
        stockList = extractor_portfolio_yahoo_sample['stockList']
        source = 'yahoo'
        for i in range(len(stockList)):
            if len(stockList[i]) == 0:
                with pytest.raises(ValueError):
                    pe = Portfolio_Extractor(stockList[i],source)
                    data = pe.get_portfolio_date(date[i][0], date[i][1])
            else:
                pe = Portfolio_Extractor(stockList[i],source)
                data = pe.get_portfolio_date(date[i][0], date[i][1])
                assert isinstance(data, Portfolio)

    def test_investing_portfolio_period(self,extractor_portfolio_investing_sample):
        period = extractor_portfolio_investing_sample['period']
        country = extractor_portfolio_investing_sample['country']
        stockList = extractor_portfolio_investing_sample['stockList']
        source = 'investing'
        for i in range(len(stockList)):
            if len(stockList[i]) == 0:
                with pytest.raises(ValueError):
                    pe = Portfolio_Extractor(stockList[i],source, country = country[i])
                    data = pe.get_portfolio_period(period[i])
            else:
                pe = Portfolio_Extractor(stockList[i],source, country = country[i])
                data = pe.get_portfolio_period(period[i])
                assert isinstance(data, Portfolio)

    def test_investing_portfolio_date(self,extractor_portfolio_investing_sample):
        date = extractor_portfolio_investing_sample['date']
        country = extractor_portfolio_investing_sample['country']
        stockList = extractor_portfolio_investing_sample['stockList']
        source = 'investing'
        for i in range(len(stockList)):
            if len(stockList[i]) == 0:
                with pytest.raises(ValueError):
                    pe = Portfolio_Extractor(stockList[i],source, country = country[i])
                    data = pe.get_portfolio_date(date[i][0], date[i][1])
            else:
                pe = Portfolio_Extractor(stockList[i],source, country = country[i])
                data = pe.get_portfolio_date(date[i][0], date[i][1])
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

# todo
class Test_base:
    
    def test_stock_make_technical_indicator(self, base_stock_test_sample):
        colNames = ['Close']
        funcList = [macd, stochastic_oscillator, rsi, williams_r]
        labels = ['macd', 'so', 'rsi', 'williams']

        outkeyList = list(base_stock_test_sample[0].keys())

        for col in colNames:
            for i in range(len(funcList)):
                base_stock_test_sample.make_technical_indicator(col, labels[i],funcList[i])
                if labels[i]=="macd":
                    outkeyList.extend(['macd_EMA_12',
                        'macd_EMA_26',
                        'macd_DIF',
                        'macd_DEM',
                        'macd_OSC'])
                elif labels[i] == 'so':
                    outkeyList.extend(['so_k_val','so_d_val'])
                else: 
                    outkeyList.append(labels[i])
        
        assert np.array_equal(list(base_stock_test_sample[0].keys()), outkeyList)
        
    def test_portfolio_get_portfolio_dataFrame(self,base_portfolio_test_sample):
        df = base_portfolio_test_sample.get_portfolio_dataFrame()
        outkeyList = ['Date','Open','High','Low','Close','Volume','Dividends','Stock Splits','symbol']
        assert np.array_equal(list(df.keys()), outkeyList)

    def test_portfolio_make_technical_indicator(self,base_portfolio_test_sample):
        colNames = ['Close']
        funcList = [macd, stochastic_oscillator, rsi, williams_r]
        labels = ['macd', 'so', 'rsi', 'williams']

        outkeyList = list(base_portfolio_test_sample.get_portfolio_dataFrame().keys())

        for col in colNames:
            for i in range(len(funcList)):
                base_portfolio_test_sample.make_technical_indicator(col, labels[i],funcList[i],1,50)
                if labels[i]=="macd":
                    outkeyList.extend(['macd_EMA_12',
                        'macd_EMA_26',
                        'macd_DIF',
                        'macd_DEM',
                        'macd_OSC'])
                elif labels[i] == 'so':
                    outkeyList.extend(['so_k_val','so_d_val'])
                else: 
                    outkeyList.append(labels[i])
        assert np.array_equal(list(base_portfolio_test_sample.get_portfolio_dataFrame().keys()), outkeyList)


    def test_portfolio_remove_different_date(self):
        se = Stock_Extractor('MSFT', 'yahoo')
        stock = se.get_stock_date('2020-06-24', '2020-07-23')
        se2 = Stock_Extractor('aapl', 'yahoo')
        stock2 = se2.get_stock_date('2020-07-01', '2020-07-23')

        pt = Portfolio([stock,stock2])
        pt.remove_different_time_index()

        assert pt.get_portfolio_dataFrame().Date.min() == '2020-07-01'
        assert pt.get_portfolio_dataFrame().Date.max() == '2020-07-23'
