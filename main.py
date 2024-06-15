import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

model = load_model('my_weights.keras')

st.header('Stock Market Predictor')
st.subheader('''This project was built by three Students from Kibo School of Technology 🏔️:\n
          [Sulaimon Olanrewaju: "KID174158864461", 
         Quawiyyah Abdulhameed: "KID495884456589",
         Goodnews Adewole: "KID102262244472"
        ]''')

start = '2012-01-01'
end = '2022-12-31'

stock =st.text_input('''Enter any of the CAC40 Stock Ticker Symbols, e.g; "GLE.PA", "MT.PA", "ENGI.PA", "RMS.PA", "RNO.PA", "HO.PA", "FR.PA" etc.''', 'AAPL')
# url = 'https://query1.finance.yahoo.com/v7/finance/download/' + stock + '?period1=' + str(start) + '&period2=' + str(end) + '&interval=1d&events=history'
# df = pd.read_csv(url)

data = yf.download(stock, start, end)

st.subheader(stock + ' Data From ' + start + ' To ' + end)
st.write(data)

# Describing Data
st.subheader(stock + ' Statistical Data')
st.write(data.describe())

#Visualisation

st.subheader('Closing Price v/s Time Chart With 100 SMA And 200 SMA')
st.write("A 100-day and 200-day Moving Average (MA) is the average of closing prices of the previous 100 days and 200 days respectively")
st.write("As Per Market Experts -> Buy signal appear when SMA-100 line cut SMA-200 line in its way upward")

ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(data.Close, 'r', label = 'Closing Price')
plt.plot(ma100, 'b', label = '100 SMA')
plt.plot(ma200, 'g', label = '200 SMA')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader(stock + ' Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(data.Close, 'r', label = 'Closing Price')
plt.plot(ma_50_days, 'g', label = '50 SMA')
plt.plot(data.Close, 'g')
plt.plot(ma_50_days, 'r')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig1)

st.subheader(stock + ' Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(data.Close, 'r', label = 'Closing Price')
plt.plot(ma_50_days, 'g', label = '50 SMA')
plt.plot(ma_100_days, 'b', label = '100 SMA')
plt.plot(data.Close, 'r')
plt.plot(ma_50_days, 'g')
plt.plot(ma_100_days, 'b')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

st.subheader(stock + ' Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(data.Close, 'r', label = 'Closing Price')
plt.plot(ma_100_days, 'g', label = '100 SMA')
plt.plot(ma_200_days, 'b', label = '200 SMA')
plt.plot(data.Close, 'r')
plt.plot(ma_100_days, 'g')
plt.plot(ma_200_days, 'b')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader(stock +' Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)

st.write("This Is Just For Educational Purpose And No Way A Financial Advice. ❤️")