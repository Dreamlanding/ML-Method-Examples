'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
Updated to Python 3.6 by Belter, Jan 6, 2019
'''
import numpy as np
import os

path = r'..\data'
training_sample = 'Logistic_Regression-trainingSample.txt'
testing_sample = 'Logistic_Regression-testingSample.txt'

# 从文件中读入训练样本的数据
def loadDataSet(p, file_n):