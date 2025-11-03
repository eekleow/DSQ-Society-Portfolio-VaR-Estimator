#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy import stats
import requests
from io import StringIO

def api_request(ticker, startDate, endDate, token):
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={startDate}&endDate={endDate}&format=csv&token={token}"
    response = requests.get(url)
    if response.status_code == 200:
        df = pd.read_csv(StringIO(response.text))
        return df
    else:
        print(f"Error: {response.status_code}")
        return None

def get_data(tickers, startDate, endDate, token):
    data = pd.DataFrame()
    for ticker in tickers:
        temp_data = api_request(ticker, startDate, endDate, token)
        if temp_data is not None:
            temp_data.set_index('date', inplace=True)
            data[ticker] = temp_data['close']
        else:
            print(f"Failed to retrieve data for {ticker}")
    return data