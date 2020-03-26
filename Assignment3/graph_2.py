from keras.models import load_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import numpy as np
import pickle
import keras
import time
import sys

model = load_model('./cnn/model2_rot.h5')

test_rot_list = pickle.load(open('./rot/rotate_2.pkl', 'rb'))
test_rot_acc = [test_rot[1] for test_rot in test_rot_list]

plot_label = 'Model2_rot'

# Rotation

plt.figure()
plt.plot(test_rot_acc, marker='o', label=plot_label)
plt.xticks(np.arange(0, 8), np.arange(-40, 40, step=10))
plt.legend(loc='upper right')
plt.ylabel('Test Accuracy')
plt.xlabel('Degree of Rotation')

plt.show()

model = load_model('./cnn/model2.h5')

test_blur_list = pickle.load(open('./blur/blurred_2.pkl', 'rb'))
test_blur_acc = [test_blur[1] for test_blur in test_blur_list]

plot_label = 'Model2'

# Blur/gaussian noise

plt.figure()

blur_list = [0.01, 0.1, 1]

plt.bar(np.arange(len(blur_list)), test_blur_acc, align='center', alpha=0.5)
plt.xticks(np.arange(len(blur_list)), blur_list)
plt.xlabel('Gaussian Noise variance')
plt.ylabel('Test accuracy')
plt.show()
