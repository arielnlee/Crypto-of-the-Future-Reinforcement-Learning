import numpy as np
import pandas as pd
import json
import requests
import ssl
import time
from datetime import datetime, timezone

# !! this code was found on cryptodatadownload's blog page and edited for the purposes of this project
# uses Binance API to download historical data 

ssl._create_default_https_context = ssl._create_unverified_context

def getUnixTimestamp(dateString, timespec='milliseconds', pattern=None):
    """
    This function takes a specific date string and tries to turn a date pattern into a unix timestamp 
    """
    if pattern is None:
        pattern = "%Y-%m-%dT%H:%M:%S"
    try:
        unix = int(time.mktime(time.strptime(dateString, pattern)))
    except ValueError as ve:
        iso_str = dateString.astimezone(timezone.utc).isoformat('T', timespec)
        unix = iso_str.replace('+00:00', 'Z')

    return unix


def OHLC_binance(pair, starttime, endtime, interval):
    """Function to read through OHLC historical timeseries 
    Accepts Binance Pair name using slash - 'ETH/USDT'
    starttime = a timestamp in this format "%Y-%m-%dT%H:%M:%S" 
    endtime = a timestamp in this format "%Y-%m-%dT%H:%M:%S"  
    interval = 1 Day (1d), 1 Hour (1h), 1 Minute (1m)
    """
    # create unix timestamps out of the date passed for Binance
    endTimeUnix = str(getUnixTimestamp(endtime)) + '000'   
    startTime = str(getUnixTimestamp(starttime)) + '000'  

    df_list = []   

    # split the slash (/) out of the symbol 'ETH/USDT' and create ETHUSDT instead
    sym = pair.split('/')    
    symbol = sym[0] + sym[1]

    while True:
        url = f"https://api.binance.com/api/v1/klines?symbol={symbol}&interval={interval}&limit=1000&startTime={startTime}&endTime={endTimeUnix}"  
        # get the request from Binance 
        response = requests.get(url)    

        # check if the response from Binance = 200 or "Good"
        if response.status_code == 200:   
            data = json.loads(response.text)  

            # take the json response and turn it into a pandas dataframe
            data_pd = pd.DataFrame(data, columns=['unix', 'open', 'high', 'low', 
                                                  'close', 'volume', 'close_unix', 'volume_from', 'tradecount', 
                                                  'marketorder_volume', 'marketorder_volume_from', 'misc'])
            # convert date out of unix timestamps
            data_pd['date'] = pd.to_datetime(data_pd['unix'], unit='ms')
            
            # drop misc column
            data_pd.drop(columns=['misc'], inplace=True)   
            data_pd['close'] = data_pd['close'].astype(str).astype(float)
            data_pd['open'] = data_pd['open'].astype(str).astype(float)
            data_pd['volume'] = data_pd['volume'].astype(str).astype(float)
            data_pd['high'] = data_pd['high'].astype(str).astype(float)
            data_pd['low'] = data_pd['low'].astype(str).astype(float)

            # append the dataframe candles to list that we are storing them in
            df_list.append(data_pd)  
            # check if len of json object is 0; if so, end loop
            if len(data_pd["unix"]) == 0:     
                break
            # reset the newest "startTime" to last date returned by previous API call
            startTime = str(data_pd['unix'].iloc[-1])  
            # since Binance returns 1000 candles, if its less than 500, end loop
            if len(data_pd["unix"]) < 500:         
                break 
        # bad server response
        else:    
            print(url)   
            print(response.status_code) 
            print(response.text)
            break

    master_df = pd.concat(df_list)  
    # drop any duplicate rows
    master_df = master_df.drop_duplicates(subset='unix', keep='first')   

    return master_df 