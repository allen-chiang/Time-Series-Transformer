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
@pytest.mark.parametrize("stockSymbol", ["aapl", "0050.TW", "GC=F"])
class Test_stock_extractor:
    @pytest.mark.parametrize("periods", ["1y", "max", "1d"])
    def test_extractor_get_stock_period(self, stockSymbol, periods,source):
        se = Stock_Extractor(stockSymbol,source)
        data = se.get_stock_period(periods)
        assert isinstance(data, Stock)

    @pytest.mark.parametrize("dates", [["2019-01-01", "2020-07-01"], ["2015-01-01", "2020-01-01"],["2020-01-01", "2020-01-01"]])
    def test_extractor_get_stock_date(self,stockSymbol, dates, source):
        se = Stock_Extractor(stockSymbol,source)
        data = se.get_stock_date(dates[0], dates[1])
        assert isinstance(data, Stock)

@pytest.mark.portoflio_extractor
@pytest.mark.parametrize("source", ["yahoo"])
@pytest.mark.parametrize("stockList", [["aapl", "0050.TW", "GC=F"],[],["aapl"]])
class Test_portfolio_extractor:
    @pytest.mark.parametrize("periods", ["1y", "max", "1d"])
    def test_extractor_get_portfolio_period(self, stockList, periods,source):
        se = Portfolio_Extractor(stockList,source)
        data = se.get_portfolio_period(periods)
        assert isinstance(data, Portfolio)

    @pytest.mark.parametrize("dates", [["2019-01-01", "2020-07-01"], ["2015-01-01", "2020-01-01"],["2020-01-01", "2020-01-01"]])
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
        # data = list([[], [100], [100,20,30], se])
        return se
    
    def test_stock_make_technical_indicator(self,stock_test_sample):
        colNames = ['Close']
        funcList = [macd, stochastic_oscillator, rsi, williams_r]
        labels = ['macd', 'so', 'rsi', 'williams']

        for col in colNames:
            for i in len(funcList):
                stock_test_sample.make_technical_indicator(col, labels[i],funcList[i])
        assert stock_test_sample.df.shape[1] == 17
        

    def test_portfolio_make_technical_indicator(self,portfolio_test_sample):
        pass

    def test_portfolio_get_portfolio_dataFrame(self,portfolio_test_sample):
        pass

    def test_portfolio_remove_different_date(self,portfolio_test_sample):
        pass

