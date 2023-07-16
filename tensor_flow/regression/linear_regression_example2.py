
# this example comes from book, "Machine Learning with Scikit-Learn and TensorFlow"
"""
对一个学习算法来说，最重要的有四个要素：
- 数据（训练数据和测试数据）
- 模型（用于预测或分类）
- 代价函数（评价当前参数的效果，对其求导可以计算梯度）
- 优化器（优化代价函数的参数，执行梯度下降）
Beter, 20170628
"""

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

housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# 方法1：使用正规方程直接求参数
def get_theta_by_normal_equation(X, y):
    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
    with tf.Session() as sess:
        theta_value = theta.eval()
        print(theta_value)

# 方法2：梯度下降法训练参数（手动求导）
def train_theta_by_gradient_descent(X, y):
    global m
    n_epochs = 1000  # 迭代次数
    learning_rate = 0.01  # 之前学习率不能太大是因为X没有做缩放
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
    y_pred = tf.matmul(X, theta, name='predictions')
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')
    gradients = 2.0/m * tf.matmul(tf.transpose(X), error)
    training_op = tf.assign(theta, theta - learning_rate * gradients)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if epoch % 10 == 0:
                print('Epoch', epoch, 'MSE =', mse.eval())
            sess.run(training_op)
        best_theta = theta.eval()
        print('Best theta is', best_theta)
# train_theta_by_gradient_descent(X_scaled, y)

# 方法3：梯度下降法训练参数（自动求导）
def train_theta_by_autodiff(X, y):
    global m
    n_epochs = 10000
    learning_rate = 0.0000003  # 学习率不能太大
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
    y_pred = tf.matmul(X, theta, name='predictions')
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')