import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from math import sin

dat = []

total_iterations = 520

for i in range(total_iterations):
    dat.append(sin(i))

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(np.array(dat).reshape(-1,1))

iterations = 100

x_train = []
y_train = []

for x in range(iterations, len(scaled_data)):
    x_train.append(scaled_data[x-iterations:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=20, batch_size=32)

########################

actual = []

for i in range(iterations):
    actual.append(sin(i+iterations))

actual = np.array(actual)

dat_df, actual_df = pd.DataFrame(data=dat), pd.DataFrame(data=actual)

total_dataset = pd.concat((dat_df, actual_df), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(actual) - iterations:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

########################

x_test = []

for x in range(iterations, len(model_inputs)):
    x_test.append(model_inputs[x-iterations:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction = model.predict(x_test)
prediction = scaler.inverse_transform(prediction)

plt.plot(actual, color='black')
plt.plot(prediction, color='red')
plt.show()