import numpy as np
#from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, MaxPool2D, Flatten
from keras.layers.convolutional import ZeroPadding2D, Conv2D
from keras import regularizers
from PIL import Image

from keras import optimizers
from keras.datasets import mnist
from keras import backend as K
from keras.models import load_model
import pickle
import keras

model = Sequential()
# CONV + POOL 1
model.add(Conv2D(filters=100,
            kernel_size=3,
            strides = 1,
            padding = 'same',
            input_shape=(256, 256, 3),
            activation='relu'
          ))
#model.add(MaxPool2D(pool_size=2, strides=2, padding='same'))

# CONV + POOL 2
model.add(Conv2D(filters=50, kernel_size=5, padding = 'same', strides=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

# CONV + POOL 3
model.add(Conv2D(filters=30,
            kernel_size=3,
            strides = 1,
            padding = 'same',
            activation='relu'
          ))

# model.add(Conv2D(filters=256,
#             kernel_size=3,
#             padding='same',
#             activation='relu'
#           ))

#Continue here

model.add(MaxPool2D(pool_size=(2, 4), strides=(2, 4)))
#
model.add(Flatten())

model.add(Dense(10, activation='softmax'))

model.summary()
