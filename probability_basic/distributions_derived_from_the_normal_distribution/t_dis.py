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
calculate_t_score()  # 2.09302405441


def calculate_ci(ci_value, data):
    """
    calculate (ci value%)-confidence interval(CI)
    :param ci_value: confidence coefficient (0, 1)
    :param data: an array
    :return: confidence interval with confidence coefficient of ci_value
    """
    df = len(data) - 1  # degrees of freedom
    ci = stats.t.interval(ci_value, df, loc=np.mean(data),
                          scale=stats.sem(data))
    return ci
norm_dis = stats.norm(0, 2)
demo_data1 = norm_dis.rvs(10)
print(demo_data1)
alpha2 = 0.95
# (-0.2217121415878075, 1.7026114809498547)
print(calculate_ci(alpha2, demo_data1))


# standard deviat