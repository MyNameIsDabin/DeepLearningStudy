import numpy as np
from mnist_dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import matplotlib.pylab as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

# 하이퍼 파라미터
iters_num = 10 # 반복 횟수
train_size = x_train.shape[0]
batch_size = 100 # 미니 배치 사이즈
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):

    print(i)
    # 미니 배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch) # 성능 개선판!

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

x = np.arange(0, iters_num)
y = np.array(train_loss_list)
plt.plot(x, y)
plt.ylim(-0.1, 5)
plt.show()