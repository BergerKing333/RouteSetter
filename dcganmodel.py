import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
from utils import *
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

def build_generator(noise_dim):
    model = Sequential()
    model.add(Dense(9*9*256, use_bias=False, input_shape=(noise_dim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    print("After Dense: ", model.output_shape)

    model.add(Reshape((9, 9, 256)))
    print("After Reshape: ", model.output_shape)

    # First upsampling layer
    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    print("After Conv2DTranspose 1: ", model.output_shape)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # Second upsampling layer
    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    print("After Conv2DTranspose 2: ", model.output_shape)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # Third upsampling layer
    model.add(Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    print("After Conv2DTranspose 3: ", model.output_shape)
    

    return model

def build_discriminator(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[36, 36, 1]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model

def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)

    return model

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    return total_loss

def train_step(images, generator, discriminator, noise_dim, batch_size):
    noise = tf.random.normal([batch_size, noise_dim])
    
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        print(f'Generator Loss: {gen_loss} \t Discriminator Loss: {disc_loss}')

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch, generator, discriminator, noise_dim, batch_size)

        print(f'Epoch {epoch} completed')

        if (epoch + 1) % 5 == 0:
            save_generated_images(epoch, generator, noise_dim)


def normalizeData(data):
    return (data - 2) / 2

def denormalizeData(data):
    return (data + 1) * 2

def save_generated_images(epoch, generator, noise_dim, examples=1):
    noise = tf.random.normal([1, noise_dim])
    # noise = noise.reshape(None, 100)
    generated_images = generator(noise, training=False)

    generated_images = tf.make_ndarray(tf.make_tensor_proto(generated_images))

    generated_images = denormalizeData(generated_images)
    generated_images = generated_images[:, :-1, :-1, :]
    img = drawImg(holdArray = generated_images[0].reshape(35, 35), saveImg=True, name=f'output.png')


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

noise_dim = 100
epochs = 10000
batch_size = 128
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

shape, holds, ratings = readCSV('training_data30.csv')
holds = normalizeData(holds).reshape(holds.shape[0], shape[0], shape[1], 1)
holds = np.pad(holds, ((0, 0), (0, 1), (0, 1), (0, 0)), mode='constant', constant_values=0)

discriminator = build_discriminator((36, 36, 1))
# discriminator.compile(optimizer=optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy', metrics=['accuracy'])

generator = build_generator(noise_dim)


train_dataset = tf.data.Dataset.from_tensor_slices(holds).shuffle(holds.shape[0]).batch(batch_size)
train(train_dataset, epochs)

discriminator.trainable = False




# gan = build_gan(generator, discriminator)
# gan.compile(optimizer=optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy')

# train_dcgan(generator, discriminator, gan, holds, noise_dim, epochs, batch_size)







# def train_dcgan(generator, discriminator, gan, dataset, noise_dim, epochs, batch_size):
#     batch_count = dataset.shape[0]
#     for epoch in range(epochs):
#         for _ in range(batch_count):
#             noise = np.random.normal(0, 1, (batch_size, noise_dim))
#             generated_images = generator.predict(noise)

#             real_images = dataset[np.random.randint(0, dataset.shape[0], batch_size)]
            
#             labels_real = np.ones((batch_size, 1))
#             labels_fake = np.zeros((batch_size, 1))

#             d_loss_real = discriminator.train_on_batch(real_images, labels_real)
#             d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)
#             d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

#             noise = np.random.normal(0, 1, (batch_size, noise_dim))
#             g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

#             print(f'Epoch: {epoch} \t Discriminator Loss: {d_loss} \t Generator Loss: {g_loss}')
#             if epoch % 10 == 0:
#                 save_generated_images(epoch, generator, noise_dim)