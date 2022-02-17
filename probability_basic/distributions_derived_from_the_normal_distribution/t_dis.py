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
    t_score = stats.t(df).isf(alpha