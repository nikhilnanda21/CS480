import numpy as np
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
import sys

num_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_dummy_train = np.zeros([60000, 32, 32])
x_dummy_test = np.zeros([x_test.shape[0], 32, 32])

print("x_dummy_train before padding:", x_dummy_train.shape)
print("x_dummy_test before padding:", x_dummy_test.shape)

for i in range(x_train.shape[0]):
    x_dummy_train[i, :, :] = np.pad(x_train[i, :, :], pad_width=2, mode='constant', constant_values=0)

for i in range(x_test.shape[0]):
    x_dummy_test[i, :, :] = np.pad(x_test[i, :, :], pad_width=2, mode='constant', constant_values=0)

print("x_dummy_train after padding:", x_dummy_train.shape)
print("x_dummy_test after padding:", x_dummy_test.shape)

x_dummy_train = x_dummy_train.reshape(x_dummy_train.shape[0], x_dummy_train.shape[1], x_dummy_train.shape[2], 1).astype('float32')
x_dummy_test = x_dummy_test.reshape(x_dummy_test.shape[0], x_dummy_test.shape[1], x_dummy_test.shape[2], 1).astype('float32')

print("x_dummy_train after reshape:", x_dummy_train.shape)
print("x_dummy_test after reshape:", x_dummy_test.shape)

print("y_train:", y_train.shape)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print("y_train after categorical:", y_train.shape)

rot_list = np.array([-40])

for rot in rot_list:
    print("Rotation : ", rot)
    X_rot = []
    for (i, X_data) in enumerate(x_dummy_train):
      sys.stdout.write("Conversion progress: %d / %d   \r" % (i, x_dummy_train.shape[0]) )
      sys.stdout.flush()
      #X = X_data.reshape(32, 32) * 255
      X = X_data.reshape(32, 32)
      img = Image.fromarray(X.astype(np.uint8), 'L')
      img_rot = img.rotate(rot)

      X_rot.append(np.array(img_rot).reshape(32, 32, 1))

    X_rot = np.array(X_rot)

for rot in rot_list:
    print("Rotation : ", rot)
    X_rot_test = []
    for (i, X_data) in enumerate(x_dummy_test):
      sys.stdout.write("Conversion progress: %d / %d   \r" % (i, x_dummy_test.shape[0]) )
      sys.stdout.flush()
      X = X_data.reshape(32, 32)
      img = Image.fromarray(X.astype(np.uint8), 'L')
      img_rot = img.rotate(rot)

      X_rot_test.append(np.array(img_rot).reshape(32, 32, 1))

    X_rot_test = np.array(X_rot_test)

print(X_rot.shape)
print(X_rot_test.shape)

model = Sequential()
# CONV + POOL 1
model.add(Conv2D(filters=64,
            kernel_size=3,
            input_shape=(32, 32, 1),
            padding='same',
            activation='relu'
          ))
model.add(MaxPool2D(pool_size=2, strides=2, padding='same'))

# CONV + POOL 2
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2, padding='same'))

# CONV + CONV + POOL 3
model.add(Conv2D(filters=256,
            kernel_size=3,
            padding='same',
            activation='relu'
          ))

model.add(Conv2D(filters=256,
            kernel_size=3,
            padding='same',
            activation='relu'
          ))

model.add(MaxPool2D(pool_size=2, strides=2, padding='same'))

# CONV + CONV + POOL 4
model.add(Conv2D(filters=512,
            kernel_size=3,
            padding='same',
            activation='relu'
          ))

model.add(Conv2D(filters=512,
            kernel_size=3,
            padding='same',
            activation='relu'
          ))

model.add(MaxPool2D(pool_size=2, strides=2, padding='same'))

# CONV + CONV + POOL 5
model.add(Conv2D(filters=512,
            kernel_size=3,
            padding='same',
            activation='relu'
          ))

model.add(Conv2D(filters=512,
            kernel_size=3,
            padding='same',
            activation='relu'
          ))

model.add(MaxPool2D(pool_size=2, strides=2, padding='same'))

# FULLY CONNECTED LAYER - 4096x2, 10x2
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

epoch = 3

sgd = optimizers.SGD(lr=0.001, momentum=0.9)

# Compile the model
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train_score_list = []
test_score_list = []
for i in range(epoch):
  train_score = model.fit(X_rot, y_train,
            batch_size=32,
            validation_split=0.2,
            initial_epoch=i,
            epochs=i+1,
            verbose=1)
  test_score = model.evaluate(X_rot_test, y_test, verbose=1)
  print("Test loss: ", test_score[0])
  print("Test accuracy: ", test_score[1])
  train_score_list.append(train_score.history)
  test_score_list.append(test_score)

# Storing the model
model.save('./cnn/model2_rot.h5')
with open('./data/train_2_rot.pkl', 'wb') as file_history:
  pickle.dump(train_score_list, file_history)

# Storing the results from training the model
with open('./data/test_2_rot.pkl', 'wb') as file_history:
  pickle.dump(test_score_list, file_history)
