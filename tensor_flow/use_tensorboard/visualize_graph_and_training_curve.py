# this example comes from book, "Machine Learning with Scikit-Learn and TensorFlow"
# 程序运行结束后，再命令行中执行tensorboard --logdir tf_logs/
# 然后再http://localhost:6006/可以查看面板

import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler  #  用于数据缩放
from datetime import datetime

# 按照时间创建存放log文件的文件夹，用于tensorBoard的可视化
now = datetime.utcnow().