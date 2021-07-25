# -*- coding: utf-8 -*-
"""
Created on Sun May 28 14:01:01 2017

@author: xin

https://gist.github.com/stewartpark/187895beb89f0a1b3a54
"""
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np

batch_size = 1
num_classes = 1
epochs = 1000

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])
x_test = np.array([[0, 0], [1, 0]])
y_test = np.array([[0], [1]])
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], '