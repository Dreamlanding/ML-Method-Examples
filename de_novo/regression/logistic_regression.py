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
    dataMat = []
    labelMat = []
    fr = open(os.path.join(p, file_n))
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 三个特征x0, x1, x2, x0=1
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))  # 样本标签y
    return dataMat, labelMat

def sigmoid(X):
    return