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
import forex_data as fd

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
path = 'C://2020_Summer//Capstone//Data//'
folders = [usd_aud, usd_cad, usd_mxn, usd_eur]

# read in data from file
dict_forex = fd.ts_data_main(path, folders)

# clean forex data
dict_clean =  fd.get_2sigma_3sigma(dict_forex,'30Min')

# helper function - returns 2,3-sigma deviation for a single stock/currency 
def get_clean_data(df, time_int):

    print("cleaning forex data...\n")
    df.index = df.Date        
    df['stddev'] = df.resample(time_int).agg({'Close':np.std})        
    df_ = df[df.stddev.notna()]        
    df_['chg_price'] = df_.Close.diff()
    df_['2-sigma'] = 2*df_.stddev.shift(1)
    df_['3-sigma'] = 3*df_.stddev.shift(1)
    #print("\n\n Last\n", df_.head())
    df_ = df_[df_.stddev.notna()]
    df_ = df_[(df_.chg_price >= df_['2-sigma']) | (df_.chg_price >= df_['3-sigma'] )]  

    return df_


# helper function -- create plot
def make_plot(df, symbol):

    print("In the plotting function\n",file=sys.stderr)
    df.rename(columns={'Date':'Date_orig'},inplace=True)
    print(df.head(), file=sys.stderr)
    data = ColumnDataSource(df)
    p = figure(x_axis_type='datetime',
             plot_height=300, plot_width=600,
             title=symbol,
             x_axis_label='Date', y_axis_label='Price',
             toolbar_location=None)
    p.line(x='Date', y='Close',source=data)
    p.add_tools(HoverTool())
    # return the plot
    return(p)


# helper function -- get data
def fetch_data(symbol, start, end, dict_forex):
    
    # get the forex currency/stock symbol           
    df = dict_forex[symbol] 
    df_clean = get_clean_data(df,'30Min')  # currenly using "30Min" -- needs to be read from user input
    df_clean = df_clean[(df_clean.Date >= start) & (df_clean.Date <= end)]
    print(df_clean.head(5), file=sys.stderr)
    
    # return dataframe
    return df_clean
	

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

    print("In the main function:", symbol, file=sys.stderr)
            
	# get dataframe 
    df = fetch_data(symbol, startDate, endDate, dict_forex)
    
	# set up plot
    fig = make_plot(df, symbol)
    script, div = components(fig)
    
	# render the page
    return render_template('home.html', script_mm=script, div_mm=div, title=header)  

'''
{{ url_for('chart_data') }}
@app.route('/chart_data')
def chart_data():
    return render_template('chart_data.html')
'''

if __name__ =='__main__':
	app.run(debug=True,use_reloader=True)
