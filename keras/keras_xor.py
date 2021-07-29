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
print(x_test.shape[0], 'test samples')

model = Sequential()  # 定义一个序贯模型

# 添加一个全连接层，输入层有2个特征，后面的隐藏层有2个单元
# 隐藏层的激活函数为tanh
# 所有层的bias都是自动添加的
model.add(Dense(2, activation='tanh', input_shape=(2,)))

# 最后的输出也是一个全连接层，激活函数为sigmoid
model.add(Dense(1, activation='sigmoid'))

# check the structure and parameters of t