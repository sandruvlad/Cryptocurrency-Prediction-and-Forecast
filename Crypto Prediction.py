from ast import Import
from cgitb import text
import dataclasses
from logging import Filter
from unittest import findTestCases
import numpy as np
from scipy.optimize import curve_fit
from importlib.resources import open_text
from PIL import Image
from turtle import title
import streamlit as st
from sympy import per
from datetime import date
import pandas as pd
import plotly.express as px
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.graph_objects as go
import time
import matplotlib.pyplot as plt
import statsmodels.api as sm
from bioinfokit.analys import stat

START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

image = Image.open(r"/Users/vlad/Downloads/569815.jpeg")

st.image(image, width=1400, caption = "Welcome to the forecasst App!")

st.markdown("#")

st.title("\nCryptocurrency Prediction - Forecast App ", )

crypto = ("BTC-GBP", "ETH-GBP", "ADA-GBP", "XRP-GBP", "SOL-GBP")
selected_coin = st.selectbox ("Please select the token: ", crypto)

st.markdown("#")

my_bar = st.progress(0)
for percent_complete in range (100):
    time.sleep(0.05)
    my_bar.progress(percent_complete + 1)

n_years = st.slider ("Please select the number of years to predict: ", 1, 5)
period = n_years * 365

df = yf.Ticker (selected_coin).history(period = "4y", interval = "1d")

df3 = df.resample("d").mean()

df3.to_csv("df3.csv")

df2 = df.resample("1d").agg({
    "Open" : "first",
    "High" : "max",
    "Low" : "min",
    "Close" : "last"
})

df2.to_csv ("test_data.csv")
df2 = pd.read_csv("test_data.csv")

df2 = df2.iloc[::-1]

df2["date"] = pd.to_datetime(df2["Date"])
df2["20wma"] = df2["Close"].rolling(window=20).mean()

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace = True, drop = False)
    return data

data_load_state = st.text ("Collecting the requested information... ")
data = load_data(selected_coin)
data_load_state.text ("Request completed, Please find the data below! ")

data.to_csv("Crypto.csv")

st.markdown("#")

st.subheader("Unfiltered Information")
st.write(df3.tail())

def data_text(df3):
    return df3.to_csv().encode ("utf-8")
csv = data_text(df3)
st.download_button(
    label = "See the unfiltered CSV File",
    data = csv,
    file_name = "Unfiltered_Crypto_Information.csv",
    mime = "text/csv",
)

def plot_unfiltered_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data["Date"], y = data["Open"], name = "Open Price"))
    fig.add_trace(go.Scatter(x = data["Date"], y = data["Open"], name = "Close Price"))
    fig.update_layout(title = "Timeplot Graph", xaxis_rangeslider_visibility = True)
    fig.update_layout(autosize = False, width = 1800, height = 600)
    st.plotly_chart(fig)

plot_unfiltered_data()

def plot_stick_data():
    fig1 = go.Figure(data= [go.Candlestick(x = df2["Date"],
        open = df2["Open"], high = df2["High"],
        low = df2["Low"], close = df2["Close"], name = "OHLC")])

    fig1.update_layout(title = "Candlestick Chart + 20 WMA", autosize = False, width = 1800, height = 600)
    fig1.update_layout(xaxis_rangeslider_visible = True, template = "plotly_dark")
    fig1.update_layout(yaxis_title = "Bitcoin Price to GBP", xaxis_title = "Date")
    fig1.add_trace(go.Scatter(x = df2["Date"], y = df2["20wma"], line = dict(color = "#0000FF"), name = "20WWMA Line"))
    st.plotly_chart(fig1)
    fig1.update_yaxes(type = "log")
    
plot_stick_data()

df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast Information")
st.write(forecast.tail())

def forecast_text(forecast):
    return forecast.to_csv().encode("utf-8")
csv = forecast_text(forecast)
st.download_button(
    label = "See the Full CSV File",
    data = csv,
    file_name ="forecast.csv",
    mime = "text/csv",
)
st.write("Cryptocurrency Prediction Based on Past Performance")
fig2 = plot_plotly(m, forecast)
fig2.update_layout(autosize = False, width = 1800, height = 600)
st.plotly_chart(fig2)

st.write("Forecast Components Based on Market Movements")
fig3 = m.plot_components (forecast, figsize = (18,12))
st.write(fig3)

st.markdown("#")
model = sm.formula.ols(formula= "Close ~ High + Low + Open", data= df)
multi_reg = model.fit()
data_load_state = st.text(multi_reg.summary())