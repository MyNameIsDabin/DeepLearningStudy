import sys, os
import numpy as np
sys.path.append(os.pardir)
sys.path.append('./example/deep-learning-from-scratch-master')
from common.util import im2col

x1 = np.random.rand(1, 3, 4, 4)
print(x1)
col1 = im2col(x1, 2, 2, stride=2, pad=0)
print(col1.shape)

print("----(im2col)---")
print(col1)
print("----(-1, 4)---")
col1 = col1.reshape(-1, 4)
print(col1)

print("----max---")
out = np.max(col1, axis=1)
print(out)

print("----reshape---")
out = out.reshape(1, 2, 2, 3)
print(out)

print("----transpose---")
out = out.transpose(0, 3, 1, 2)
print(out)