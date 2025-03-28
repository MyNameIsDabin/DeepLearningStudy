import sys, os

sys.path.append(os.pardir)
import numpy as np
from mnist_dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape) # (60000, 784) 784=28x28
print(t_train.shape) # (60000, 10)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) # 0이상, train_size 미만 수 중 무작위 batchsize 만큼 뽑기
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]