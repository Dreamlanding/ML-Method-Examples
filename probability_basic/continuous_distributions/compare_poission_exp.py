import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def compare_poission_exp():
    """
    This post explained the relation between these two distribution
      - https://stats.stackexchange.com/a/2094/134555
      - P(Xt <= x) = 1 - e^(-lambda * x)
    Now, I suppose lambda=1, just like this example(from wiki, Poisson_distribution):
      - On a particular river, overflow floods occur once every 100 years on average.
    :return:
    """
    x = np.arange(20)
    y1 = 1 - np.power(np.e, -x)  # lambda = 1
    y2 = 1 - np.p