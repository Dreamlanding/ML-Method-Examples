# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

"""
Created on Sun Nov 17 18:44:37 2017

@author: Belter
"""


def sampling2pmf(n, dist, t=100000):
    """
    n: sample size for each experiment
  