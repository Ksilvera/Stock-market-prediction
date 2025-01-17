import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

#get Data 
stock_data = fetch_stock_data("AAPL", "2010-01-01", "2020-12-31")

#Predict next day's closing price
stock_data['Prediction'] = stock_data['Close'].shift(-1)

#Drop the last row
stock_data.dropna(inplace=True)

#Define the feature and target
X = stock_data[['Close']] #Feature input is the current day's close price
y = stock_data['Prediction'] #Target is the next day's close price

#Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#Create the model
model = LinearRegression()
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
