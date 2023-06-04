# this example comes from standford's cs224d lecture7, tensorflow tutorial
# https://cs224d.stanford.edu/lectures/
import numpy as np
# import seaborn
import matplotlib.pyplot as plt
import tensorflow as tf


# define input data
X_data = np.arange(100, step=.1)
y_data = X_data + 20 * np.sin(X_data/10)

# define data size and batch size
n_samples = 1000
batch_size = 100

# tensorflow is finicky about shapes, so resize
X_data = np.reshape(X_data, (n_samples, 1))
y_data = np.reshape(y_data, (n_samples, 1))

# define placeholders for input
X = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))

# define variables to be learned
with tf.variable_scope('linear-regression'):
    W = tf.get_variable('weights', (1, 1), initializer=tf.random_normal_initializer())
    b = tf.get_variable('bias', (1,), initializer=tf.constant_initializer(0.0))
    y_pred = tf.matmul(X, W) + b
    loss = tf.reduce_sum((y - y_pred)**2 / n_samples)

# sample code to run one step of gradient descent
opt = tf.train.AdamOptimizer()
opt_operation = opt.minimize(loss)

best_para = {'W':0, 'b':0, 'loss_val':100}
with tf.Session() as sess:
    # init