import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

@st.cache_resource
def load_model_cached(model_path):
    return load_model(model_path)

@st.cache_data
def load_data(stock, start, end):
    return yf.download(stock, start=start, end=end, progress=False)

model = load_model_cached('my_weights.keras')

st.header('Stock Market Predictor')
st.subheader('''This project was built by three Students from Kibo School of Technology 🏔️:\n
          [Sulaimon Olanrewaju: "KID174158864461", 
         Quawiyyah Abdulhameed: "KID495884456589",
         Goodnews Adewole: "KID102262244472"
        ]''')

start = '2012-01-01'
end = '2022-12-31'

stock =st.text_input('''Enter any of the CAC40 Stock Ticker Symbols, e.g; "GLE.PA", "MT.PA", "ENGI.PA", "RMS.PA", "RNO.PA", "HO.PA", "FR.PA" etc.''', 'AAPL')

data = load_data(stock, start, end)

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

@st.cache_resource
def create_plot(data, ma100, ma200):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(data, 'r', label='Closing Price')
    plt.plot(ma100, 'b', label='100 SMA')
    plt.plot(ma200, 'g', label='200 SMA')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    return fig

fig = create_plot(data.Close, ma100, ma200)
st.pyplot(fig)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader(stock + ' Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
# fig1 = plt.figure(figsize=(8,6))
fig1 = create_plot(data.Close, ma_50_days, ma_50_days)
st.pyplot(fig1)

st.subheader(stock + ' Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
# fig2 = plt.figure(figsize=(8,6))
fig2 = create_plot(data.Close, ma_50_days, ma_100_days)
st.pyplot(fig2)

st.subheader(stock + ' Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
# fig3 = plt.figure(figsize=(8,6))
fig3 = create_plot(data.Close, ma_100_days, ma_200_days)
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