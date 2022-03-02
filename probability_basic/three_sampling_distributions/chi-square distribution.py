
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def chi2_distribution(df=1):
    """
    卡方分布，在实际的定义中只有一个参数df，即定义中的n
    :param df: 自由度，也就是该分布中独立变量的个数
    :return:
    """