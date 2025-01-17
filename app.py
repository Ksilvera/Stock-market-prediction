import yfinance as yf
import pandas as pd
import numpy as np
import talib as ta
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def add_technical_indicators(df):
    #Moving averages
    df['MA_50'] = ta.SMA(df['Close'], timeperiod=50) #50-day moving average
    df['MA_200'] = ta.SMA(df['Close'], timeperiod=200) #200-day moving average

    #Relative Strength Index (RSI)
    df['RSI'] = ta.RSI(df['Close'], timeperiod=14)

    #Moving Average Convergence Divergence (MACD)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    #Bollinger Bands
    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = ta.BBANDS(df['Close'], timeperiod=20)

    #Average True Range (ATR)
    df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    #Drop NA values
    df.dropna(inplace=True)

    return df

#get Data 
stock_data = fetch_stock_data("AAPL", "2010-01-01", "2020-12-31")

stock_data = add_technical_indicators(stock_data)

#Predict next day's closing price
stock_data['Prediction'] = stock_data['Close'].shift(-1)

#Drop the last row
stock_data.dropna(inplace=True)

#Define the feature and target
X = stock_data[['Close', 'MA_50', 'MA_200', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'ATR']] #Feature input is the current day's close price
y = stock_data['Prediction'] #Target is the next day's close price

#Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#Create the model
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

#Make predictions
predictions = model.predict(X_test)

#Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

#Plot the data
plt.figure(figsize=(10,6))
plt.plot(y_test.index, y_test, label='Actual Price', color='blue')
plt.plot(y_test.index, predictions, label='Predicted Price', color='red')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction: Actual vs Predicted')
plt.legend()
plt.show()
