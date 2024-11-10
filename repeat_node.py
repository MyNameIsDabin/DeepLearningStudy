import numpy as np
# D, N = 8, 7
# x = np.random.randn(N, D)
# y = np.sum(x, axis=0, keepdims=True)

# dy = np.random.randn(1, D)
# dx = np.repeat(dy, N, axis=0)

# print(dy)
# print(dx)

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.matmul(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        self.grads[0][...] = dW

        return dx
    

