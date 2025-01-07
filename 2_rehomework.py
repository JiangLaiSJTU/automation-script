import numpy as np
import matplotlib.pyplot as plt
from typing import Type, Dict, Any, Callable
from math import sqrt, pi, e
from scipy.integrate import quad
import seaborn as sns

def exp(lamda, size):# 指数函数
    return np.random.exponential(1 / lamda, size)

def poisson(p, size):# 泊松分布
    return np.random.poisson(p, size)

def gaussian(miu, sigma, size):# 高斯分布
    return np.random.normal(miu, sigma, size)

def uniform(low, high, size):# 均匀分布
    return np.random.uniform(low, high, size)

def binomial(p, n, size):# 二项分布
    return np.random.binomial(n, p, size)

def xsum_P(array, avr, var, x:float):
    """
    :param array: np.array, size: times * n是一个从右往左建立的过程
    :param x:
    """
    times, n = array.shape
    sum = np.sum(array, axis=1)
    sum = (sum-n*avr)/sqrt(n*var)

    num=0
    for arr in sum:
        if arr<=x:
            num+=1
    return num/times

def x_sum_normalized(array, avr, var):
    times, n = array.shape
    sum = np.sum(array,axis=1)
    sum = (sum-n*avr)/sqrt(n*var)
    return sum

def x_sum(array):
    sum = np.sum(array, axis=1)
    return sum

def gaussian_normal_calculus(low, high):
    def gaussian_normal(x):
        return 1/sqrt(2*pi)*e**(-0.5*x*x)

    return quad(gaussian_normal, low, high)

def find_p(array, x):
    num=len(array)
    sum=0
    for i in array:
        if i<=x:

            sum+=1
    return sum/num

arr = np.random.poisson(1, (10,2))
print(arr)
sum = np.sum(arr,axis=1)
print(sum)
