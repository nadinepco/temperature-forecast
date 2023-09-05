import seaborn as sns
import pandas as pd
import numpy as np

# data visualization stack
import matplotlib as mpl
import matplotlib.pyplot as plt

# stationarity
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller

# statistics stack
from statsmodels.stats.diagnostic import het_white
from statsmodels.formula.api import ols

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def rolling_mean_plot(series,years):
    """"""
    window = int(years * 365.24)
    rolling = series.rolling(window).mean()
    rolling.plot(legend=True)
    sns.despine()

def resampled_mean_plot(series, years):
    """"""
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    resampled = s.resample(str(years)+'Y').mean()
    resampled.plot(legend=True);
    sns.despine()

def qcd_variance(series,window=12):
    """
    This function returns the quartile coefficient of dispersion
    of the rolling variance of a series in a given window range 
    """
    # rolling variance for a given window 
    variances = series.rolling(window).var().dropna()
    # first quartile
    Q1 = np.percentile(variances, 25)#, interpolation='midpoint')
    # third quartile
    Q3 = np.percentile(variances, 75)#, interpolation='midpoint')
    # quartile coefficient of dispersion 
    qcd = round((Q3-Q1)/(Q3+Q1),6)
    
    print(f"quartile coefficient of dispersion: {qcd}")

def create_lagged_features(df, number_of_lags):
    """"""
    df = df[['remainder']]
    
    lags = list(range(1, number_of_lags+1))
    
    for lag in lags:
        column_name = 'lag_' + str(lag)
        df[column_name] = df['remainder'].shift(lag)
        
    return df

# homoscedasticity test
def white_homoscedasticity_test(series):
    """
    returns p-value for White's homoscedasticity test
    """
    series = series.reset_index(drop=True).reset_index()
    series.columns = ['time', 'value']
    series['time'] += 1
    
    olsr = ols('value ~ time', series).fit()
    p_value = het_white(olsr.resid, olsr.model.exog)[1]
    
    return round(p_value,6)

# stationarity test p-values
def p_values(series):
    """
    returns p-values for ADF and KPSS Tests on a time series
    """
    # p value from Augmented Dickey-Fuller (ADF) Test
    p_adf = adfuller(series, autolag="AIC")[1]
    
    # p value from Kwiatkowski–Phillips–Schmidt–Shin (KPSS) Test
    p_kpss = kpss(series, regression="c", nlags="auto")[1]
    
    return round(p_adf,6), round(p_kpss,6)

# function for stationarity test
def test_stationarity(series):
    """
    returns likely conclusions about series stationarity
    """
    # test homoscedasticity
    p_white = white_homoscedasticity_test(series)
    
    if p_white < 0.05:
        print(f"\n non-stationary: heteroscedastic (White test p-value: {p_white}) \n")
    
    # test stationarity
    else:
        p_adf, p_kpss = p_values(series)
        
        # print p-values
        print( f"\n p_adf: {p_adf}, p_kpss: {p_kpss}" )
    
        if (p_adf < 0.05) and (p_kpss >= 0.05):
            print('\n stationary or seasonal-stationary')
            
        elif (p_adf >= 0.1) and (p_kpss < 0.05):
            print('\n difference-stationary')
            
        elif (p_adf < 0.1) and (p_kpss < 0.05):
            print('\n trend-stationary')
        
        else:
            print('\n non-stationary; no robust conclusions\n')

def auto_correlation_plot(series):
    """
    plots autocorrelations for a given series
    """
    mpl.rc('figure',figsize=(10,2),dpi=200)
    plot_acf(series,zero=False,lags=25)
    plt.xlabel('number of lags')
    plt.ylabel('autocorrelation')

def partial_auto_correlation_plot(series):
    """
    plots partial autocorrelations for a given series
    """
    mpl.rc('figure',figsize=(10,2),dpi=200)
    plot_pacf(series,zero=False,lags=25)
    plt.xlabel('number of lags')
    plt.ylabel('partial autocorrelation')

def check_stationarity_adf(series):
    # Copied from https://machinelearningmastery.com/time-series-data-stationary-python/

    result = adfuller(series.values)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon-stationary\x1b[0m")

def create_lagged_features(df, number_of_lags):
    """"""
    df = df[['remainder']]
    
    lags = list(range(1, number_of_lags+1))
    
    for lag in lags:
        column_name = 'lag_' + str(lag)
        df[column_name] = df['remainder'].shift(lag)
        
    return df