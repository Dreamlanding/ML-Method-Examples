# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

"""
Created on Sun Nov 12 08:44:37 2017

@author: Belter
"""


def sampling2pmf(n, dist, m=100000):
    """
    n: sample size for each experiment
    m: how many times do you do experiment, fix in 100000
    dist: frozen distribution
    """
    current_dist = dist
    sum_of_samples = []
    for i in range(m):
        samples = current_dist.rvs(size=n)  # 与每次取一个值，取n次效果相同
        # print(samples)
        sum_of_samples.append(np.sum(