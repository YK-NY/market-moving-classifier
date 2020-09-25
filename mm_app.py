from flask import Flask, render_template, request
from bokeh.embed import components
from bokeh.models import HoverTool
import pandas as pd
import numpy as np
import requests
import sys

from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook
from bokeh.models import DatetimeTickFormatter
from bokeh.models import ColumnDataSource

import sys
sys.path.append('C://2020_Summer//Capstone//modules//')

import modules.forex_data as fd
import modules.mm_model as mm

from datetime import datetime

# flask object
app = Flask(__name__)

# dir name for each currency type
usd_eur = 'USD_EUR'
usd_cad = 'USD_CAD'
usd_mxn = 'USD_MXN'
usd_aud = 'USD_AUD'

# path for the data
global path
global folders
global dict_forex
global dict_clean
global df_tweets
global df_kmc
path = 'C://2020_Summer//Capstone//deploy//Data//'
folders = [usd_aud, usd_cad, usd_mxn, usd_eur]
#folders = [usd_cad, usd_mxn, usd_eur]

# read in data from file
dict_forex = fd.ts_data_main(path, folders)

# clean forex data
# dict_clean =  fd.get_2sigma_3sigma(dict_forex,'30Min')

# get tweets
df_tweets = fd.get_tweets(path)

# identify tweets
dict_clean = fd.identify_tweets(df_tweets, dict_forex)


# train and test set
df_train, df_test = mm.get_train_test(df_tweets)

# get clustered data from kmeans
# df_kmc = mm.get_kmeans_model(df_train,n_clusters=5)
df_kmc = pd.read_json('C://2020_Summer//Capstone//deploy//df_kmc.json')

# create data for model
df_final = fd.create_data_for_model(dict_clean)  


# classification
df_predicted = mm.classification_model(df_final)



# helper function - returns 2,3-sigma deviation for a single stock/currency 
def get_clean_data(df, time_int):
    
    df.index = df.Date        
    df['stddev'] = df.resample(time_int).agg({'Close':np.std})        
    df_ = df[df.stddev.notna()]        
    df_['chg_price'] = df_.Close.diff()
    df_['2-sigma'] = 2*df_.stddev.shift(1)
    df_['3-sigma'] = 3*df_.stddev.shift(1)    
    df_ = df_[df_.stddev.notna()]
    df_ = df_[(df_.chg_price >= df_['2-sigma']) | (df_.chg_price >= df_['3-sigma'] )]  

    return df_


# helper function -- create plot
def make_plot(df, symbol):

    #print("In the plotting function\n",file=sys.stderr)
    df.rename(columns={'tweets_date':'Date_orig'},inplace=True)
    
    data = ColumnDataSource(df)
    p = figure(x_axis_type='datetime',
             plot_height=300, plot_width=600,   
             title=symbol,             
             x_axis_label='Date', y_axis_label='Price',
             toolbar_location=None)
    p.title.align="center"
    p.line(x='Date', y='Close',source=data)    
    #p.add_tools(HoverTool())   
    p.add_tools(HoverTool(tooltips=[("Date", '@Date_orig{%a%d - %H:%M}'),("Price", '@Close'),("Tweet",'@text')], formatters={'@Date_orig':'datetime'}, mode='vline'))    
    
    # return the plot
    return(p)


# helper function -- get data
def fetch_data(symbol, start, end, dict_clean):    
    
    # get the forex currency/stock symbol           
    df = dict_clean[symbol] 
    
    st_dt = datetime.strptime(start, '%Y-%m-%d')
    end_dt = datetime.strptime(end, '%Y-%m-%d')    
    
    df = df[(df.tweets_date.dt.year >= st_dt.year) & (df.tweets_date.dt.year <= end_dt.year)]    

    df.target.fillna(0,inplace=True)
    df_1 = df[df.target == 1]
    df_0 = df[df.target == 0]    
    
    # return dataframe
    return df_1, df_0
	

@app.route('/')
def homepage():
    return render_template('Home.html',debug=True)

# reads user input from form
# gets data for the selected stock/forex
# plot chart
# render on webpage
@app.route('/',methods=['POST'])
def user_input():
    header="Forex Chart"
    startDate = request.form['startdate']
    endDate = request.form['enddate']
    symbol = request.form['currency']    
    #values = request.form.getlist('features')

    #print("In the main function:", symbol, file=sys.stderr)
            
	# get dataframe 
    df_1, df_0 = fetch_data(symbol, startDate, endDate, dict_clean)
    
    df_1_tweets = df_1
    df_1_tweets.reset_index(drop=True,inplace=True)
    indices = df_1_tweets.index.tolist()
    # get similar tweets - returns a list
    cluster_tweets = mm.get_clustered_tweets(indices,df_kmc)  
	
    # set up plot
    fig = make_plot(df_1, symbol)
    script, div = components(fig)    
        
    # get tweets
    
    df_pos = df_1[['Date_orig','text']]
    df_neg = df_0[['tweets_date','text']]
    
    df_pred = df_predicted[df_predicted.target == 1]        
    df_pred = df_pred[['tweets_date','text']]
    
	# render the page
    return render_template('Home.html', script_mm=script, div_mm=div, title=header, div_tweets_cluster=cluster_tweets,
                           div_tweets_1=df_pos.to_html(header=False,index=False, col_space=10,border='',justify='unset'),
                           div_tweets_2=df_neg.to_html(header=False,index=False, col_space=10,border='',justify='unset'),
                           div_tweets_pred=df_pred.to_html(header=False,index=False, col_space=10,border='',justify='unset'))  

if __name__ =='__main__':
	app.run(debug=True,use_reloader=True)