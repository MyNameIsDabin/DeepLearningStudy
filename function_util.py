import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # 넘파이 배열을 반환하고 정수와 더하는 과정에서 브로드 캐스트

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y