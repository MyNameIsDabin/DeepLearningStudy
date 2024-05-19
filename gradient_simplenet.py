import sys, os

sys.path.append(os.pardir)
import numpy as np

from function_util import softmax, cross_entropy_error, numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.array([[0.47355232, 0.9977393, 0.84668094], 
                           [0.85557411, 0.03563661, 0.69422093]]) # np.random.randn(2, 3) # 2x3 정규 분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = np.dot(x, self.W) # length: 3
        y = softmax(z) # length: 3
        loss = cross_entropy_error(y, t) # length : 1
        return loss
    
net = simpleNet()
x = np.array([0.6, 0.9]) # 1 x 2
t = np.array([0, 0, 1]) # 1 x 3

def f(W):
    return net.loss(x, t) # x: [0.6, 0.9], t: [0, 0, 1], 0.92806853663411326

dW = numerical_gradient(f, net.W)
print(dW)