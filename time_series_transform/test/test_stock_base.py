import pytest
from time_series_transform.stock_transform.base import *
from time_series_transform.stock_transform.stock_extractor import *
from time_series_transform.stock_transform.util import *

# test marks
#   - stock_extractor
#   - portfolio_extractor
#   - util
#   - base 
# Quck command to run mark: pytest -v -m <mark>

@pytest.mark.stock_extractor
@pytest.mark.parametrize("source", ["yahoo"])
@pytest.mark.parametrize("stockSymbol", ["aapl", "0050.TW", "MSFT"])
class Test_stock_extractor:
    @pytest.mark.parametrize("periods", ["1y", "max", "1d"])
    def test_extractor_get_stock_period(self, stockSymbol, periods,source):
        se = Stock_Extractor(stockSymbol,source)
        data = se.get_stock_period(periods)
        assert isinstance(data, Stock)

    @pytest.mark.parametrize("dates", [["2020-01-01", "2020-07-01"], ["2020-01-01", "2020-03-01"],["2020-01-01", "2020-02-01"]])
    def test_extractor_get_stock_date(self,stockSymbol, dates, source):
        se = Stock_Extractor(stockSymbol,source)
        data = se.get_stock_date(dates[0], dates[1])
        assert isinstance(data, Stock)

@pytest.mark.portoflio_extractor
@pytest.mark.parametrize("source", ["yahoo"])
@pytest.mark.parametrize("stockList", [["aapl", "0050.TW", "MSFT"],[],["aapl"]])
class Test_portfolio_extractor:
    @pytest.mark.parametrize("periods", ["1y", "max", "1d"])
    def test_extractor_get_portfolio_period(self, stockList, periods,source):
        se = Portfolio_Extractor(stockList,source)
        data = se.get_portfolio_period(periods)
        assert isinstance(data, Portfolio)

    @pytest.mark.parametrize("dates", [["2020-06-01", "2020-07-01"], ["2020-01-01", "2020-02-01"],["2020-01-01", "2020-01-01"]])
    def test_extractor_get_portfolio_date(self,stockList, dates, source):
        pe = Portfolio_Extractor(stockList,source)
        data = pe.get_portfolio_date(dates[0], dates[1])
        assert isinstance(data, Portfolio)


@pytest.mark.util
class Test_stock_util:

    @pytest.fixture(scope = 'class')
    def arr(self):
        se = Stock_Extractor('aapl','yahoo').get_stock_period('1y').df['Close']
        data = list([[], [100], [100,20,30], se])
        return data


    @pytest.mark.parametrize("return_diff", [True,False])
    def test_macd(self, arr, return_diff):
        for ar in arr:
            macd_res = []
            oriLen = len(ar)
            if oriLen == 0:
                with pytest.raises(ValueError):
                    macd_res = macd(ar, return_diff)
            else:
                macd_res = macd(ar, return_diff)
                outKeys = []
                
                if return_diff:
                    assert len(macd_res) == oriLen
                else:
                    col = list(macd_res.keys())
                    outKeys = ['EMA_12', 'EMA_26', 'DIF', 'DEM', 'OSC']
                    assert np.array_equal(col, outKeys)
                    for key in outKeys:
                        assert len(macd_res[key].shape) == 1
                        assert macd_res[key].shape[0] == oriLen 


    def test_stochastic_oscillator(self, arr):
        for ar in arr:
            so_res = []
            oriLen = len(ar)
            if oriLen == 0:
                with pytest.raises(ValueError):
                    so_res = stochastic_oscillator(ar)
            else:
                so_res = stochastic_oscillator(ar)
                col = list(so_res.keys())
                outKeys = ['k_val', 'd_val']
                assert np.array_equal(col, outKeys)
                for key in outKeys:
                    assert len(so_res[key].shape) == 1
                    assert so_res[key].shape[0] == oriLen

    def test_rsi(self, arr):
        for ar in arr:
            rsi_res = []
            oriLen = len(ar)
            if oriLen == 0:
                with pytest.raises(ValueError):
                    rsi_res = rsi(ar)
            else:
                rsi_res = rsi(ar)
                assert len(rsi_res.shape) == 1
                assert rsi_res.shape[0] == oriLen

    def test_williams_r(self, arr):
        for ar in arr:
            w_res = []
            oriLen = len(ar)
            if oriLen == 0:
                with pytest.raises(ValueError):
                    w_res = williams_r(ar)
            else:
                w_res = williams_r(ar)
                assert len(w_res.shape) == 1
                assert w_res.shape[0] == oriLen

@pytest.mark.base
class Test_base:
    @pytest.fixture(scope = 'class')
    def stock_test_sample(self):
        se = Stock_Extractor('aapl','yahoo').get_stock_period('1y')
        return se

    @pytest.fixture(scope = 'class')
    def portfolio_test_sample(self):
        stockList = ["aapl", "0050.TW", "MSFT"]
        pe = Portfolio_Extractor(stockList,'yahoo').get_portfolio_period('1y')
        return pe
    
    def test_stock_make_technical_indicator(self,stock_test_sample):
        colNames = ['Close']
        funcList = [macd, stochastic_oscillator, rsi, williams_r]
        labels = ['macd', 'so', 'rsi', 'williams']

        outkeyList = list(stock_test_sample.df.keys())

        for col in colNames:
            for i in range(len(funcList)):
                stock_test_sample.make_technical_indicator(col, labels[i],funcList[i])
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
        
        assert np.array_equal(list(stock_test_sample.df.keys()), outkeyList)
        
    def test_portfolio_get_portfolio_dataFrame(self,portfolio_test_sample):
        df = portfolio_test_sample.get_portfolio_dataFrame()
        outkeyList = ['Date','Open','High','Low','Close','Volume','Dividends','Stock Splits','symbol']
        assert np.array_equal(list(df.keys()), outkeyList)

    def test_portfolio_make_technical_indicator(self,portfolio_test_sample):
        colNames = ['Close']
        funcList = [macd, stochastic_oscillator, rsi, williams_r]
        labels = ['macd', 'so', 'rsi', 'williams']

        outkeyList = list(portfolio_test_sample.get_portfolio_dataFrame().keys())

        for col in colNames:
            for i in range(len(funcList)):
                portfolio_test_sample.make_technical_indicator(col, labels[i],funcList[i],1,50)
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
        assert np.array_equal(list(portfolio_test_sample.get_portfolio_dataFrame().keys()), outkeyList)


    def test_portfolio_remove_different_date(self):
        se = Stock_Extractor('MSFT', 'yahoo')
        stock = se.get_stock_date('2020-06-24', '2020-07-23')
        se2 = Stock_Extractor('aapl', 'yahoo')
        stock2 = se2.get_stock_date('2020-07-01', '2020-07-23')

        pt = Portfolio([stock,stock2])
        pt.remove_different_date()

        assert pt.get_portfolio_dataFrame().Date.min() == '2020-07-01'
        assert pt.get_portfolio_dataFrame().Date.max() == '2020-07-23'
