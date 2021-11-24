import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def uniform_distribution(loc=0, scale=1):
    """
    均匀分布，在实际的定义中有两个参数，分布定义域区间的起点和终点[a, b]
    :param loc: 该分布的起点, 相当于a
    :param scale: 区间长度, 相当于 b-a
    :return:
    """
    uniform_dis = stats.uniform(loc=loc, scale=scale)
    x = np.linspace(uniform_dis.ppf(0.01),
                    uniform_dis.ppf(0.99), 100)
    fig, ax = plt.subplots(1, 1)

    