# Stock-Prediction-of-Amazon-using-LSTM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load the historical stock price data
data = pd.read_csv('AMZN.csv')
prices = data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

# Create a time series dataset
look_back = 30  # Number of previous days' prices to consider
X, y = [], []
for i in range(len(prices_scaled) - look_back):
    X.append(prices_scaled[i:i+look_back, 0])
    y.append(prices_scaled[i+look_back, 0])
X, y = np.array(X), np.array(y)

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape input data for LSTM (samples, time steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions to get actual stock prices
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train[0], train_predict[:, 0]))
test_rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:, 0]))

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# Plot the results
plt.plot(scaler.inverse_transform(prices_scaled), label='True')
plt.plot(np.arange(look_back, look_back + len(train_predict)), train_predict, label='Train Predict')
plt.plot(np.arange(look_back + len(train_predict), len(prices_scaled)), test_predict, label='Test Predict')
plt.legend()
plt.show()
