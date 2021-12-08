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

    # 直接传入参数
    ax.plot(x, stats.uniform.pdf(x, loc=2, scale=4), 'r-',
            lw=5, alpha=0.6, label='uniform pdf')

    # 从冻结的均匀分布取值
    ax.plot(x, uniform_dis.pdf(x), 'k-',
            lw=2, label='frozen pdf')

    # 计算ppf分别等于0.001, 0.5, 0.999时的x值
    vals = uniform_dis.ppf([0.001, 0.5, 0.999])
    print(vals)  # [ 2.004  4.     5.996]

    # Check accuracy of cdf and ppf
    print(np.allclose([0.001, 0.5, 0.999], uniform_dis.cdf(vals)))  # Ture

    r = uniform_dis.rvs(size=10000)
    ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
    plt.ylabel('Probability')
    plt.title(r'PDF of Unif({}, {})'.format(loc, loc+scale))
    ax.legend(loc='best', frameon=False)
    plt.show()

# uniform_distribution(loc=2, scale=4)


def exponential_dis(loc=0, scale=1.0):
    """
    指数分布，exponential continuous random variable
    按照定义，指数分布只有一个参数lambda，这里的scale = 1/lambda
    :param loc: 定义域的左端点，相当于将整体分布沿x轴平移loc
    :param scale: lambda的倒数，loc + scale表示该分布的均值，scale^2表示该分布的方差
    :return:
    """
    exp_dis = stats.expon(loc=loc, scale=scale)
    x = np.linspace(exp_dis.ppf(0.000001),
                    exp_dis.ppf(0.999999), 100)
    fig, ax = plt.subplots(1, 1)

    # 直接传入参数
    ax.plot(x, stats.expon.pdf(x, loc=loc, scale=scale), 'r-',
            lw=5, alpha=0.6, label='uniform pdf')

    # 从冻结的均匀分布取值
    ax.plot(x, exp_dis.pdf(x), 'k-',
            lw=2, label='frozen pdf')

    # 计算ppf分别等于0.001, 0.5, 0.999时的x值
    vals = exp_dis.ppf([0.001, 0.5, 0.999])
    print(vals)  # [ 2.004  4.     5.996]

    # Check accuracy of cdf and ppf
    print(np.allclose([0.001, 0.5, 0.999], exp_dis.cdf(vals)))

    r = exp_dis.rvs(size=10000)
    ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
    plt.ylabel('Probability')
    plt.title(r'PDF of Exp(0.5)')
    ax.legend(loc='best', frameon=False)
    plt.show()

# exponential_dis(loc=0, scale=2)


def diff_exp_dis():
    """
    不同参数下的指数分布
    :return:
    """
    exp_dis_0_5 = stats.expon(scale