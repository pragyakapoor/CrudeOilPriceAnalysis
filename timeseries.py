# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 20:48:15 2019

@author: Pragya Kapoor
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data.csv')
dataset['Date'] = pd.to_datetime(dataset['Date'], infer_datetime_format=True)
indexedDataset = dataset.set_index(['Date'])

from datetime import datetime
indexedDataset.tail(5)

plt.xlabel('Date')
plt.ylabel('Crude Oil Prices')
plt.plot(indexedDataset)

#determining rolling statistics
rolmean = indexedDataset.rolling(window=365).mean()
rolstd = indexedDataset.rolling(window=365).std()
print(rolmean, rolstd)

#Plot rolling statistics
orig = plt.plot(indexedDataset, color='blue',label='Original')
mean = plt.plot(rolmean , color='red', label = 'Rolling Mean')
std= plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling mean and Standard Deviation')
plt.show()

#Perform Dickey-Fuller test : 
from statsmodels.tsa.stattools import adfuller

print('Results of Dickey-Fuller test:')
dftest = adfuller(indexedDataset['Value'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags used', 'Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)

#Estimating trend
indexedDataset_logscale = np.log(indexedDataset)
plt.plot(indexedDataset_logscale)      

movingAverage = indexedDataset_logscale.rolling(window=365).mean()
movingSTD = indexedDataset_logscale.rolling(window=365).std()
plt.plot(indexedDataset_logscale)
plt.plot(movingAverage,color='red')

datasetLogScaleMinusMovingAverage = indexedDataset_logscale - movingAverage
datasetLogScaleMinusMovingAverage.head(12)

#removing nan values
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(10)        

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determining rolling statistics
    movingAverage = timeseries.rolling(window=365).mean()
    movingSTD = timeseries.rolling(window=365).std() 

    #Plot Rolling Statistics
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(movingAverage , color='red', label = 'Rolling Mean')
    std= plt.plot(movingSTD, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling mean and Standard Deviation')
    plt.show(block=False)
    
    #Perform Dicky-Fuller Test:
    print('Results of Dickey-Fuller test:')
    dftest = adfuller(timeseries['Value'], autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags used', 'Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(datasetLogScaleMinusMovingAverage) 

exponentialDecayWeightedAverage = indexedDataset_logscale.ewm(halflife=12, min_periods=0,adjust=True).mean()
plt.plot(indexedDataset_logscale)
plt.plot(exponentialDecayWeightedAverage, color='red')

datasetLogScaleMinusMovingExponentialDecayAverage = indexedDataset_logscale-exponentialDecayWeightedAverage
test_stationarity(datasetLogScaleMinusMovingExponentialDecayAverage)

datasetLogDiffShifting = indexedDataset_logscale - indexedDataset_logscale.shift()
plt.plot(datasetLogDiffShifting)

datasetLogDiffShifting.dropna(inplace=True)
test_stationarity(datasetLogDiffShifting)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(indexedDataset_logscale,freq=2000)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(indexedDataset_logscale, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residual')
plt.legend(loc='best')
plt.tight_layout()

decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(datasetLogDiffShifting, nlags = 20)
lag_pacf = pacf(datasetLogDiffShifting, nlags=20, method='ols')

#plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--', color='gray')
plt.title('Autocorrelation Function')

#plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

from statsmodels.tsa.arima_model import ARIMA

#AR Model
model = ARIMA(indexedDataset_logscale, order=(3,1,2))
results_AR = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4F'% sum((results_AR.fittedvalues-datasetLogDiffShifting["Value"])**2))
print('plotting AR model')

model= ARIMA(indexedDataset_logscale, order=(3,1,1))
results_ARIMA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues,color='red')
plt.title('RSS: %4F'% sum(results_ARIMA.fittedvalues-datasetLogDiffShifting["Value"]**2))

prediction_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(prediction_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum = prediction_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

predictions_ARIMA_log = pd.Series(indexedDataset_logscale['Value'].ix[0], index=indexedDataset_logscale.index)
predictions_Arima_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA_log.head()

predictions_ARIMA= np.exp(predictions_ARIMA_log)
plt.plot(indexedDataset)
plt.plot(predictions_ARIMA)

indexedDataset_logscale

results_ARIMA.plot_predict(1,7977)
x=results_ARIMA.forecast(steps=3650)
results_ARIMA.forecast(steps=120)




                     
