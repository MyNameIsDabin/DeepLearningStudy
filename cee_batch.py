import numpy as np

def cross_entropy_error_datum(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

# 원-핫 인코딩인 경우
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

# 원-핫 인코딩이 아닌 경우 (숫자 레이블)
def cross_entropy_error_label(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0] # 2

    return -np.sum(np.log(y[[np.arange(batch_size)], t] + 1e-7)) / batch_size

# 정답 2
t = [2, 
     2]

# # 예1 '2'일 확률이 가장 높다고 추정함 (0.6)
y = [[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0], 
     [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]] #2x10

print(cross_entropy_error_label(np.array(y), np.array(t)))

#y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
#print(cross_entropy_error(np.array(y), np.array(t)))

a = np.array([[1, 2], [3, 4]])
print(a[[0, 1], [1, 1]])