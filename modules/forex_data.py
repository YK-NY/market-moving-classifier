# read in forex data
import glob
import re
import os

import pandas as pd
import numpy as np

usd_eur = 'USD_EUR'
usd_cad = 'USD_CAD'
usd_mxn = 'USD_MXN'
usd_aud = 'USD_AUD'


path = 'C://2020_Summer//Capstone//Data//'
folders = [usd_aud, usd_cad, usd_mxn, usd_eur]


# takes path for the files
# returns a dataframe

def read_files(path, folder):
    
    os.chdir(path+folder+'//data')    
    files = glob.glob('*.csv')    
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f,header=None))            
    
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.columns = ['Date','Time','Open','High','Low','Close','Vol']
    os.chdir('../..')
    print("folder:",folder, "end of reading files..:",os.getcwd() )
    
    return df_all

# format data -- use right datatypes, keep necessary columns
def format_data(df):
    # convert date and time columns to datetime object
    df.Date = pd.to_datetime(df.Date +' '+df.Time, errors='coerce')
    
    # drop the Time and other columns
    # keep only date and the closing price
    df.drop(columns=['Time', 'Open','High','Low','Vol'], inplace=True)
    
    # sort by ascending
    df.sort_values(['Date'], inplace=True, ignore_index=True)        
    
    # average for 15 min
    return df
    

# read data and format
# input: takes a path (string) and a list of strings for folders
# outputs: a dictionary of clean dataframe for each forex for all the years data is available

def ts_data_main(path, folders):
    dict_forex = {}

    print("forex_data.. reading and formating")
    
    for f in folders:    
        df = read_files(path, f)
        dict_forex.update({f:format_data(df)})

    os.chdir('../')
    print("after reading and formatting data...:",os.getcwd() )
    
    return dict_forex 

# input: dictionary of forex time series data and the required time interval: 15,30,60min
# output: dicitionary of clean df, filtered for prices that exhibits a 2, 3-sigma deviation in the specified time window

def get_2sigma_3sigma(dict_forex, time_int):
    # takes time series data for a stock/currency
    # compute std dev at 15, 30 min, 60 min intervals, based on input
    # retain original price info at each of these intervals.
    # If the price is > 2 or 3-sigma deviations at that time, retain them and filter the rest
    # return this time series df
    
    # compute std_dev and save results as an additional column
    
    dict_clean = {}
    for key, items in dict_forex.items():
        #print(key)
        df = dict_forex[key]        
        df.index = df.Date
        
        df['stddev'] = df.resample(time_int).agg({'Close':np.std})
        #df['mean']
        df_ = df[df.stddev.notna()]        
        df_['chg_price'] = df_.Close.diff()
        df_['2-sigma'] = 2*df_.stddev.shift(1)
        df_['3-sigma'] = 3*df_.stddev.shift(1)
        #print("\n\n Last\n", df_.head())
        df_ = df_[df_.stddev.notna()]
        df_ = df_[(df_.chg_price >= df_['2-sigma']) | (df_.chg_price >= df_['3-sigma'] )]
        dict_clean.update({key:df_})
    
    return dict_clean
           
