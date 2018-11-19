# -*- coding: utf-8 -*-


import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import warnings
from alpha_vantage.timeseries import TimeSeries
import sys

warnings.filterwarnings("ignore")




def get_ticker_data(time_series_func, ticker_symbol, inter):
    
    
    ts = TimeSeries(key='INSERT KEY HERE', output_format='pandas')
    try:
        if time_series_func == "intraday":    
            df, meta_data = ts.get_intraday(symbol=ticker_symbol, interval=inter, outputsize='full')
        elif time_series_func == "daily":
            df, meta_data = ts.get_daily_adjusted(symbol=ticker_symbol, outputsize='full')
        elif time_series_func == "weekly":
            df, meta_data = ts.get_weekly_adjusted(symbol=ticker_symbol, outputsize='full')
        elif time_series_func == "monthly":
            df, meta_data = ts.get_monthly_adjusted(symbol=ticker_symbol, outputsize='full')
    except:
        raise ValueError('Syntax error, please check your arguments, e.g. (intraday, MSFT, 5min etc...)')
    
    df = df[["4. close"]]
    df.index = pd.to_datetime(df.index)
    
    return df


def aic_param(df):
    
    # Function which finds the best p,d,q fit through 
    # Akaike Information Criterion (AIC) trial and error.
    

    p = d = q = 0
    pdq = aic = []
    
    for p in range(5):
        for d in range(2):
            for q in range(5):
                try:
                    model = ARIMA(df, (p, d, q)).fit()
                    x = model.aic
                    x1 = p,d,q
    
                    aic.append(x)
                    pdq.append(x1)
                    
                except:
                    pass
                    # ignore the error and go on
                  
    pdq_index = aic.index(min(aic))
    return pdq[pdq_index]
    

            
            
def ARIMA_predict(df, interval, difference_interval, time_steps):
    
    # python produces an extra time step.
    time_steps = int(time_steps)-1
    
    difference_interval = int(difference_interval)
    
    
    # difference interval to shift the dataset, will probably try to change this
    # in the future.
    #intraday = {"1min": 1, "5min": 1, "15min": 1, "30min": 1, "60min": 1}
    
    timedelta = {"1min": "1 min", "5min": "5 min", "15min": "15 min", 
                 "30min": "30 min", "60min": "60 min", "daily": "1 days", 
                 "weekly": "7 days", "monthly": "30 days"}


    # invert differenced value
    def difference_inverse(history, yhat, difference_interval):
        
    	return yhat + history[-difference_interval]
            
    X = df.values

    differenced = np.diff(df.values, difference_interval, 0)
    
    # fit model
    model = ARIMA(differenced, order=aic_param(differenced))
    model_fit = model.fit(disp=0)
    
    # multi-step out-of-sample prediction
    start_index = len(differenced)
    end_index = start_index + time_steps
    prediction = model_fit.predict(start=start_index, end=end_index)
    # invert the differenced prediction to something usable
    history = [x for x in X]
    step = 1
    
    for yhat in prediction:
    	inverted = difference_inverse(history, yhat, difference_interval)
    	print('Time_step %d: %f' % (step, inverted))
    	history.append(inverted)
    	step += 1
    
    df_length = len(df)
    for i in range(time_steps):
        
        # Create the row to be added
        row = pd.Series({"4. close": history[-(i+1)][0]}, name = df.index[df_length+i-1] +
                                             pd.Timedelta(timedelta.get(interval)))
        
        df = df.append(row)
          
    df.to_csv("%s_" % sys.argv[2]+"%s_" % sys.argv[1] + "%s_" % sys.argv[5] + "time_steps.csv")






if __name__ == "__main__":
    # arg1 = time series function (e.g. intraday, weekly etc..)
    # arg2 = ticker symbol (MSFT, AMZN etc...)
    # arg3 = interval (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
    # arg4 = difference_interval
    # arg5 = time_steps
    df = get_ticker_data(sys.argv[1], sys.argv[2], sys.argv[3])
    
    ARIMA_predict(df, sys.argv[3], sys.argv[4], sys.argv[5])
    
    # EXAMPLE 1: python ARIMA3.py intraday MSFT 1min 1 10
    # EXAMPLE 2: python ARIMA.py daily AMZN daily 1 7
