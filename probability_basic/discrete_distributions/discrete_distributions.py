# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 18:47:10 2017

@author: xin
"""

# an example
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def example1():
    # 分布的参数初始化
    myDF = stats.norm(5, 3)  # Create the frozen distribution
    # 取101个等间距的x
    X = np.linspace(-5, 15, 101)
    # cdf, 累计分布函数
    y = myDF.cdf(X)  # Calculate the corresponding CDF
    plt.plot(X, y)


def bernoulli_distribution():
    # 伯努利分布
    # 只有一个参数：p，实验成功的概率
    p = 0.6
    bernoulli_dist = stats.bernoulli(p)

    # 伯努利分布的概率质量分布函数pmf
    p_heads = bernoulli_dist.pmf(1)  # 试验结果为1的概率, 规定为正面, 概率为0.6
    p_tails = bernoulli_dist.pmf(0)  # 试验结果为0的概率, 规定为反面, 概率为0.4

    # 取100个服从参数为0.6的伯努利分布的随机变量
    trials = bernoulli_dist.rvs(100)

    print(np.sum(trials))  # 63, 相当于1的个数

    # 100个随机变量的直方图, 相当于取出来的100个随机变量的概率质量分布
    plt.hist(trials/len(trials))
    # plt.show()
    plt.savefig('bernoulli_pmf.png', dpi=200)
    plt.close()

    # 0-2