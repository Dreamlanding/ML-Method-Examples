# this example comes from book, "Machine Learning with Scikit-Learn and TensorFlow"

import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler  #  用于数据缩放

housing = fetch_california_housing()
m, n = housing.data.shape  # m是样本数，n是特征的数量
print(m, n)

# Gradient Descent requires scaling the feature vectors first
# X的缩放对后面的训练过程影响非常大，经过缩放的数据经过很少的迭代次数就可以收敛，学习率可以设得很大
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
X_scaled = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X_scaled')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')

# 方法2：梯度下降法训练参数（手