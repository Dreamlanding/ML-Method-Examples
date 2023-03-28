
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:26:08 2017

@author: Belter
"""

# 广义线性模型
# 基本假设：目标值是由输入变量的线性组合得到的
# http://scikit-learn.org/stable/modules/linear_model.html

#------------------------ part1: 系数由普通最小二乘法估计得到--------------------
# http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
# 基本假设：变量之间是相互独立的
from sklearn import linear_model
reg = linear_model.LinearRegression()
X = [[0, 0], [1, 1], [2, 2]] # Training data, 形状：[n_samples, n_features]
y = [0, 1, 2]  # Target values,  形状：[n_samples, n_targets]
reg.fit(X, y)  # 训练
# 由下面两个参数可以得到拟合的直线方程为 y = 0.5*x_1 + 0.5*x_2
reg.coef_  # 系数的值， array([ 0.5,  0.5])
reg.intercept_  #  截距， 2.2204460492503131e-16
reg.predict([8, 9])  # 预测一个新的x的值，array([ 8.5])


#---------- 一个更复杂的例子
print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

