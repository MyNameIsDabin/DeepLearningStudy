import numpy as np
import matplotlib.pylab as plt

# 나쁜 구현의 예
# def numerical_diff(f, x):
#     h = 1e-50
#     return (f(x + h) - f(x)) / h


# 문제 1 : 반올림 오차 발생 (너무 작은 수를 컴퓨터로 올바르게 표현할 수 없음)
# 문제 2 : f의 차분.(함수 결과의 차이)


def numerical_diff(f, x):
    h = 1e-4 # 대충 이정도로 10의 -4승 값 정도로 좋은 결과를 얻는다고 알려졌다 
    return (f(x + h) - f(x - h)) / (2 * h)


def function_1(x):
    return 0.01*x**2 + 0.1*x

# print(numerical_diff(function_1, 5))
# print(numerical_diff(function_1, 10))

# x = np.arange(0.0, 20.0, 0.1)
# y = function_1(x)
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.plot(x, y)
# plt.show()


def function_2(x):
    return x[0]**2 + x[1]**2 # np.sum(x**2)

# x0=3, x1=4일 때, x0에 대한 편미분을 구하라.
def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

print(numerical_diff(function_tmp1, 3.0))

def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

print(numerical_diff(function_tmp2, 4.0))

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # x와 형상이 같은 배열 생성
    # ex) [0, 0]

    for idx in range(x.size):
        tmp_val = x[idx] # 3 or 4
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

print(numerical_gradient(function_2, np.array([3.0, 4.0])))