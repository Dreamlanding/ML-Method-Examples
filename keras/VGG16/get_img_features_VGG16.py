
# https://keras-cn.readthedocs.io/en/latest/other/application/
from keras.models import Sequential
from keras.models import Model
from keras.layers import (Flatten, Dense, Conv2D, MaxPooling2D)
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import os
import argparse
import json


def vgg16_model(weights_path):
    # this modle totaly has 22 layers with polling 
    model = Sequential()
    # Block 1
    # 这里隐藏了每一层的深度维，本来应该是(3, 3, 3)
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', 
                     input_shape=(224, 224, 3), name='block1_conv1'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
