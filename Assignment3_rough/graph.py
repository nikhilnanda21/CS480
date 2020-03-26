from keras.models import load_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import numpy as np
import pickle
import keras
import time
import sys

train_loss_list = []
train_acc_list = []

test_loss_list = []
test_acc_list = []

model = load_model('./cnn/model1.h5')

train_history_list = pickle.load(open('./data/train.pkl', 'rb'))
test_history_list = pickle.load(open('./data/test.pkl', 'rb'))

print(type(train_history_list))
print(train_history_list)

train_loss_list = [train_history['loss'] for train_history in train_history_list]
train_acc_list = [train_history['accuracy'] for train_history in train_history_list]

test_loss_list = [test_history[0] for test_history in test_history_list]
test_acc_list = [test_history[1] for test_history in test_history_list]

# 4.2 (a), (b)
plt.figure()
plt.title('Model:1, Loss Graph, Epoch 1-5')
plt.plot(train_loss_list, marker='o', label='Training Set')
plt.plot(test_loss_list, marker='o', label='Test Set')
plt.xticks(np.arange(0, 5), np.arange(1, 7))
plt.legend(loc='upper right')
plt.ylabel('Loss')
plt.xlabel('Epoch')

# 4.2 (c), (d)
plt.figure()
plt.title('Model1 , Accuracy Graph, Epoch 1-5')
plt.plot(train_acc_list, marker='o', label='Training Set')
plt.plot(test_acc_list, marker='o', label='Test Set')
plt.xticks(np.arange(0, 5), np.arange(1, 7))
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

#Rotation

test_rot_list = pickle.load(open('./rot/rotate.pkl', 'rb'))
test_rot_acc = [test_rot[1] for test_rot in test_rot_list]

plot_label = 'Model1'

plt.figure()
plt.plot(test_rot_acc, marker='o', label=plot_label)
plt.xticks(np.arange(0, 8), np.arange(-40, 40, step=10))
plt.legend(loc='upper left')
plt.ylabel('Test Accuracy')
plt.xlabel('Degree of Rotation')

# Gaussian Noise

test_blur_list = pickle.load(open('./blur/blurred.pkl', 'rb'))
test_blur_acc = [test_blur[1] for test_blur in test_blur_list]

plot_label = 'Model1'

blur_list = [0.01, 0.1, 1]

plt.figure()
plt.bar(np.arange(len(blur_list)), test_blur_acc, align='center', alpha=0.5)
plt.xticks(np.arange(len(blur_list)), blur_list)
plt.xlabel('Gaussian Noise variance')
plt.ylabel('Test accuracy')
plt.show()
