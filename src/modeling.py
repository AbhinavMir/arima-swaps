import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the CSV data into a Pandas DataFrame
df = pd.read_csv('../data/raw/uniswap_data_sample.csv')

# Convert the UnixTimestamp column to a datetime object
df['DateTime'] = pd.to_datetime(df['DateTime'], unit='s')

# Set the DateTime column as the DataFrame index
df.set_index('DateTime', inplace=True)

# Group the DataFrame by day and sum the Value_IN column
daily_volume = df['Value_IN(ETH)'].resample('D').sum()

# Plot the daily trading volume
daily_volume.plot(figsize=(12, 6))
plt.title('Daily Uniswap Trading Volume')
plt.xlabel('Date')
plt.ylabel('ETH Volume')
plt.show()

# Split the data into training and test sets
train_size = int(len(daily_volume) * 0.8)
train, test = daily_volume[:train_size], daily_volume[train_size:]

# Fit an ARIMA model to the training data
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()

# Make predictions on the test data
predictions = model_fit.forecast(len(test))[0]

# Calculate the mean squared error of the predictions
mse = mean_squared_error(test, predictions)
print(f'MSE: {mse}')

# Plot the actual and predicted values
plt.plot(test.index, test.values, label='Actual')
plt.plot(test.index, predictions, label='Predicted')
plt.title('ARIMA Predictions of Daily Uniswap Trading Volume')
plt.xlabel('Date')
plt.ylabel('ETH Volume')
plt.legend()
plt.show()
