import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('training_data.csv')

x = df.drop('rating', axis=1)
x = x.drop('Unnamed: 0', axis=1)
y = df['rating']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)


model = Sequential([
    Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])


model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

history = model.fit(x_train, y_train, epochs=100, batch_size=50, validation_data=(x_test, y_test))

loss, accuracy = model.evaluate(x_test, y_test)

outcome = model.predict(x_test)[0] - y_test.values[:]

print(y_test.values[:])

print(outcome)

print(np.mean(np.abs(outcome)))

print(loss, accuracy)