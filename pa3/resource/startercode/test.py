import numpy as np

N = 5
C = 4
y = [0,1,2,3,1]
labels_onehot = np.zeros([N, C])
labels_onehot[np.arange(N), y] = 1.0

print(labels_onehot)


# 设置类别的数量

# 需要转换的整数
# 将整数转为一个10位的one hot编码
print(np.eye(C)[y])
