import os
import sys
import numpy as np
import matplotlib.pylab as plt

X = np.array([[51, 55], [14, 19], [0, 4]])

Y = X.flatten()

#print(Y[np.array([1, 2])])

#print(Y[np.array([True, False, True, False, False, False])])


# np.reshape, np.nditer, np.zeros_like, np.pad, np.zeros, np.random.randn
# 변수명 앞에 *, **
# 표준 편차, mean, 
# sys.path.append(os.pardir)
# np.random.choice(train_size, batch_size)
# np.arange(0, 6, 0.1) # 0 ~ 6 까지 0.1 간격으로 생성

#print(os.pardir)

x = np.arange(-1.1, 1.1, 0.1)
y1 = np.exp(x)
y2 = np.exp(-x)
y3 = 1+np.exp(-x)
y4 = 1/(np.exp(-x))
y5 = 1/(1+np.exp(-x))


plt.xlabel("x")
plt.ylabel("exp(x)")
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.plot(x, y4)
plt.plot(x, y5)
plt.show()