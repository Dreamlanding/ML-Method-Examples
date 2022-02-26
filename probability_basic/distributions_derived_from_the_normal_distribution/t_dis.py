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


# standard deviation vs standard error of the mean(SEM)
# SEM: 平均值标准误差
a = [69, 54, 80]
b = [47, 68, 52]
print(np.std(a), np.std(b))  # 样本的标准差
print(stats.sem(a), stats.sem(b))  # 样本均值的标准差(对SEM的估计)
# 10.6562449088 8.95668589503
# 7.53510303697 6.33333333333


def calculate_p_value(data, ref_value)