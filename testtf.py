import tensorflow as tf


print(tf.__version__)

print(len(tf.config.experimental.list_physical_devices('GPU')))