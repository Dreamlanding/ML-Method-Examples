# this example comes from book, "Machine Learning with Scikit-Learn and TensorFlow"

import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler  #  用于数据缩放

housing = fetch_california_housing()
m, n = housing.data.shape  # m是样本数，n是特征的数量
print(