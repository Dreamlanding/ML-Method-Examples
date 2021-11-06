import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def compare_poission_exp():
    """
    This post explained the relation between these two distribution
      - https://stats.stackexchange.com/a/2094/134555
      - P(Xt <= x) = 1 - e^(-lambda * x)
    Now, I suppose lambda=1