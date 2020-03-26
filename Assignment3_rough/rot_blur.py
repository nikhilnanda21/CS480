from keras.models import load_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import numpy as np
import pickle
import keras
import time
import sys

model = load_model('./cnn/model1.h5')

_, (X_test, y_test) = mnist.load_data()

X_dummy_test = np.zeros([X_test.shape[0], 32, 32])

for i in range(X_test.shape[0]):
    X_dummy_test[i, :, :] = np.pad(X_test[i, :, :], pad_width=2, mode='constant', constant_values=0)

X_dummy_test = X_dummy_test.reshape(X_dummy_test.shape[0], X_dummy_test.shape[1], X_dummy_test.shape[2], 1).astype('float32')

y_test = keras.utils.to_categorical(y_test, 10)

rot_list = np.arange(-40, 40, step=10)

test_score_list = []

# Rotate test images

for rot in rot_list:
    print("Rotation : ", rot)
    X_rot = []
    for (i, X_data) in enumerate(X_dummy_test):
      sys.stdout.write("Conversion progress: %d / %d   \r" % (i, X_dummy_test.shape[0]) )
      sys.stdout.flush()
      X = X_data.reshape(32, 32)
      img = Image.fromarray(X.astype(np.uint8), 'L')
      img_rot = img.rotate(rot)

      X_rot.append(np.array(img_rot).reshape(32, 32, 1))

    X_rot = np.array(X_rot)

    print("\nConversion Done")
    test_score = model.evaluate(X_rot, y_test, verbose=1)
    print("Test loss: ", test_score[0])
    print("Test accuracy: ", test_score[1])
    print()

    test_score_list.append(test_score)

# Store the results
with open('./rot/rotate.pkl', 'wb') as file_history:
  pickle.dump(test_score_list, file_history)



# Blur/add gaussian noise to test images

test_score_list = []

blur_list = np.array([0.01, 0.1, 1])

for blur in blur_list:
    print("Gaussian Noise : ", blur)
    X_blur = []
    for (i, X_data) in enumerate(X_dummy_test):
      sys.stdout.write("Conversion progress: %d / %d   \r" % (i, X_dummy_test.shape[0]) )
      sys.stdout.flush()
      X = X_data.reshape(32, 32)
      img = Image.fromarray(X.astype(np.uint8), 'L')
      img_blur = img.filter(ImageFilter.GaussianBlur(blur))

      X_blur.append(np.array(img_blur).reshape(32, 32, 1))

    X_blur = np.array(X_blur)

    print("\nConversion Done")
    test_score = model.evaluate(X_blur, y_test, verbose=1)
    print("Test loss: ", test_score[0])
    print("Test accuracy: ", test_score[1])
    print()

    test_score_list.append(test_score)

# Store the results
with open('./blur/blurred.pkl', 'wb') as file_history:
  pickle.dump(test_score_list, file_history)
