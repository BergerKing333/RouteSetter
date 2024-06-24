import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.read_csv('training_data.csv')

x = df.drop('rating', axis=1)
x = x.drop('Unnamed: 0', axis=1)
y = df['rating']

x_train, x_test, y_train, y_test_1 = train_test_split(x, y, test_size=0.05)

x_train = x_train.values.reshape(-1, 476, 1)
x_test = x_test.values.reshape(-1, 476, 1)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test_1, num_classes=10)

model = models.Sequential([
    layers.Conv1D(32, 3, activation='relu', input_shape=(476, 1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(128, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

optimizer = optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=20, batch_size=50, validation_data=(x_test, y_test))

loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

# outcome = model.predict(x_test)[0] - y_test.values[:]

outcome = model.predict(x_test)

# print(y_test.values[:])

print(np.argmax(outcome, axis = 1))
print(np.argmax(y_test, axis = 1))
# print(y_test_1)

# print(outcome.astype(int))

# print(np.mean(np.abs(outcome)))


print(loss, accuracy)