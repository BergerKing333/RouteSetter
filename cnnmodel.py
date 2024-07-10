import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import pandas as pd
from utils import readCSV
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# df = pd.read_csv('training_data.csv')

# x = df.drop('rating', axis=1)
# x = x.drop('Unnamed: 0', axis=1)
# y = df['rating']

shape, x, y = readCSV('training_data30.csv')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

x_train = x_train.reshape(x_train.shape[0], shape[0], shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], shape[0], shape[1], 1)

# y_train = tf.keras.utils.to_categorical(y_train, num_classes=12)
# y_test = tf.keras.utils.to_categorical(y_test_1, num_classes=12)

model = models.Sequential([
    layers.Conv2D(35, (3, 3), activation='relu', padding='same', input_shape=(shape[0], shape[1], 1)),
    layers.BatchNormalization(),
    # layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    # layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), strides=(2, 2)),
    layers.Dropout(0.4),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    # layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    # layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), strides=(2, 2)),
    layers.Dropout(0.4),

    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    # layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    # layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), strides=(2, 2)),
    layers.Dropout(0.4),

    layers.Flatten(),

    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(1, activation='linear')
    # layers.Dense(12, activation='softmax')
])

optimizer = optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mae', metrics=['accuracy'])

checkpoint = tf.keras.callbacks.ModelCheckpoint('model.keras', monitor='val_loss', save_best_only=True)

history = model.fit(x_train, y_train, epochs=70, batch_size=50, validation_data=(x_test, y_test), callbacks=[checkpoint])

loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

# outcome = model.predict(x_test)[0] - y_test.values[:]

outcome = model.predict(x_test)
outcome = np.round(outcome, 5).reshape(-1)

# print(np.argmax(outcome, axis = 1) - np.argmax(y_test, axis = 1))

# print(np.average(np.argmax(outcome, axis = 1) - np.argmax(y_test, axis = 1)))

# print(outcome)
# print(y_test)

# print(np.array([outcome - y_test], dtype=int)[0])
# print(y_test_1)

print(np.round([outcome - y_test], 2))
print(np.average(np.abs(outcome-y_test)))
print(np.median(np.abs(outcome-y_test)))

plt.hist(outcome - y_test, bins=30, density=True)
plt.xlabel('Difference')
plt.ylabel('Density')
plt.title('Outcome - y_test')
plt.show()
# print(np.array([outcome.reshape(-1) - y_test], dtype=int)[0])
# print(np.mean(np.abs(outcome)))


print(loss, accuracy)