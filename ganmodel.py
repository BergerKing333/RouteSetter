import numpy as np
import pandas as pd
import drawRoute
import tensorflow as tf
from tensorflow.keras import layers

def normalizeInput(data):
    return (data - 2) / 2

def denormalizeOutput(data):
    return (data + 1) * 2

def build_generator(latent_dim):
    model = tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_dim=latent_dim),
        layers.Dense(512, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(476, activation='tanh')
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Dense(1024, activation='relu', input_dim=476),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_gan(generator, discriminator):
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.trainable = False
    
    gan_input = tf.keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    return gan

def train_gan(generator, discriminator, gan, data, epochs=12000, batch_size=64, latent_dim=100):
    half_batch = int(batch_size / 2)

    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], half_batch)
        real_samples = data[idx]
        real_labels = np.ones((half_batch, 1))

        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_samples = generator.predict(noise)
        fake_labels = np.zeros((half_batch, 1))

        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_y)

        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, D Loss: {d_loss[0]}, D Accuracy: {d_loss[1]}, G Loss: {g_loss}')

routes = pd.read_csv('training_data30.csv')
routes = routes.drop('Unnamed: 0', axis=1)
routes = routes.drop('rating', axis=1)
routes = normalizeInput(routes.to_numpy())

latent_dim = 100
generator = build_generator(latent_dim)
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

train_gan(generator, discriminator, gan, routes)



noise = np.random.normal(0, 1, (1, latent_dim))
generated_route = generator.predict(noise)

img = drawRoute.drawImg(holdArray=denormalizeOutput(generated_route[0]))

generator.save('generator.h5')
generator.save('gen2.keras')