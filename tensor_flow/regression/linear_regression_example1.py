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

# tensorflow is fini