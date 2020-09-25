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


path = 'C://2020_Summer//Capstone//deploy//Data//'
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
    
    for f in folders:    
        df = read_files(path, f)
        dict_forex.update({f:format_data(df)})

    os.chdir('../')    
    
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
        df = dict_forex[key]        
        df.index = df.Date
        
        df['stddev'] = df.resample(time_int).agg({'Close':np.std})        
        df_ = df[df.stddev.notna()]        
        df_['chg_price'] = df_.Close.diff()
        df_['2-sigma'] = 2*df_.stddev.shift(1)
        df_['3-sigma'] = 3*df_.stddev.shift(1)        
        df_ = df_[df_.stddev.notna()]
        df_ = df_[(df_.chg_price >= df_['2-sigma']) | (df_.chg_price >= df_['3-sigma'] )]
        dict_clean.update({key:df_})
    
    return dict_clean

# get tweets
def get_tweets(path):
    os.chdir(path)
    df = pd.read_csv('realdonaldtrump.csv')
    
    url_pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    pic_pattern = re.compile('pic\.twitter\.com/.{10}')
    df['text'] = df['content'].apply(lambda x: url_pattern.sub('',x))
    df['text'] = df['text'].apply(lambda x: pic_pattern.sub('', x))

    # removing space in mentions and hashtags
    df['text'] = df['text'].apply(lambda x: x.replace('@ ', '@'))
    df['text'] = df['text'].apply(lambda x: x.replace('# ', '#'))
    

    # get length of the tweet
    df['len'] = df['content'].str.split().map(lambda x: len(x))

    # keep only tweets greater than a word
    df = df[df.len > 1]

    # convert to datetime obj
    df['date'] = pd.to_datetime(df['date'],errors='coerce')    
    df_clean = df[['date','text']]

    os.chdir('../')

    return df_clean

# identify tweets that had an impact on financial instruments
def identify_tweets(df,dict_forex):
    dict_currency = {}
    # set index for tweets ds
    df.rename(columns={'date':'tweets_date'},inplace=True)
    df.tweets_date = df.tweets_date.dt.floor('Min')
    df.index = df.tweets_date
    
    for key,items in dict_forex.items():
        cur = dict_forex[key]
        cur.index = cur.Date
        df_ = pd.merge(df, cur, left_index=True, right_index=True, how='outer')
        df_ = df_[df_.tweets_date.notna()]
        df_['stddev'] = df_.resample('15Min').agg({'Close':np.std})     # the time interval can be read from the user input   
        #df_ = df_tweets[df_tweets.stddev.notna()]        
        df_['chg_price'] = df_.Close.diff()
        df_['2-sigma'] = 2*df_.stddev.shift(1)
        df_['3-sigma'] = 3*df_.stddev.shift(1)
        
        #df_ = df_[df_.stddev.notna()]
        mask = ((df_.chg_price >= df_['2-sigma']) | (df_.chg_price >= df_['3-sigma']))        
        df_.loc[mask,'target'] = 1
        
        dict_currency.update({key:df_})
    #return updated tweets df, dictionary of tweets dfs for each currency
    
    return dict_currency

# prepare tweets data for the classification model
# takes as input processed ts data for currencies
def create_data_for_model(dict_curr):
    
    df_list = []
    df_full_list = []
    for key, items in dict_curr.items():
        df_temp = dict_curr[key]
        df_full_list.append(df_temp)        
        df_temp = df_temp[df_temp.target == 1]
        
        df_list.append(df_temp[['tweets_date','text','Close','target']])    
    
    df_all = pd.concat(df_list,axis=0)
    df_full = pd.concat(df_full_list,axis=0)
    df_1 = df_full[df_full.target == 1]
    df_0 = df_full[df_full.target !=1] 
    
    # drop duplicated rows in df with target=Nan 
    df_0 = df_0[~df_0.index.duplicated(keep='first')]
    df_updated = pd.concat([df_0,df_1],axis=0)

    cols_drop = ['Date','stddev','2-sigma','3-sigma','chg_price']

    df_updated.drop(columns=cols_drop,inplace=True)
    df_updated.target.fillna(0,inplace=True) # replace Nan with 0.
    
    #df_updated.text = df_updated.text.apply(lambda x: x.replace('....',''))
    #df_updated.text = df_updated.text.apply(lambda x: x.replace('...',''))
    
    return df_updated
    
    

    
    
           