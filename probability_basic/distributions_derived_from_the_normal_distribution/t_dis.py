import numpy as np
from scipy import stats


def calculate_t_score():
    """
    计算上alpha分位数值，相当于已知某个点的上分位数(1-cdf)，求对于的t score
    :return:
    """
    n = 20
    df = n - 1  # 自由度
    alpha = 0.05
    t_score = stats.t(df).isf(alpha/2)  # 相当于计算t_0.025, 上0.025分位数
    print(t_score)
calculate_t_score()  # 2.09302