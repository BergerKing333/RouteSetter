import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import pandas as pd
from drawRoute import readCSV

from sklearn.model_selection import train_test_split

# df = pd.read_csv('training_data.csv')

# x = df.drop('rating', axis=1)
# x = x.drop('Unnamed: 0', axis=1)
# y = df['rating']

shape, x, y = readCSV('training_data30.csv')

x_train, x_test, y_train, y_test_1 = train_test_split(x, y, test_size=0.05)

x_train = x_train.reshape(x_train.shape[0], shape[0], shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], shape[0], shape[1], 1)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=12)
y_test = tf.keras.utils.to_categorical(y_test_1, num_classes=12)

model = models.Sequential([
    layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(shape[0], shape[1], 1)),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), strides=(2, 2)),
    layers.Dropout(.7),

    layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), strides=(2, 2)),
    layers.Dropout(.7),

    layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), strides=(2, 2)),
    layers.Dropout(.7),

    layers.Flatten(),

    layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.9),
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.9),

    layers.Dense(12, activation='softmax')
])

optimizer = optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=250, batch_size=50, validation_data=(x_test, y_test))

loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

# outcome = model.predict(x_test)[0] - y_test.values[:]

outcome = model.predict(x_test)

# print(y_test.values[:])

print(np.argmax(outcome, axis = 1) - np.argmax(y_test, axis = 1))

print(np.average(np.argmax(outcome, axis = 1) - np.argmax(y_test, axis = 1)))
# print(y_test_1)

# print(outcome.astype(int))

# print(np.mean(np.abs(outcome)))


print(loss, accuracy)