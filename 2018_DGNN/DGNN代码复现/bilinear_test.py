import torch
import torch.nn as nn
import numpy as np

m = nn.Bilinear(20, 30, 40)
input1 = torch.randn(128, 20)
input2 = torch.randn(128, 30)
output = m(input1, input2)
print(output.shape)

arr_output = output.data.cpu().numpy()

weight = m.weight.data.cpu().numpy()  # [40, 20, 30]
bias = m.bias.data.cpu().numpy()  # [40,]
x1 = input1.data.cpu().numpy()  # [128, 20]
x2 = input2.data.cpu().numpy()  # [128, 30]

y = np.zeros((x1.shape[0],weight.shape[0]))  # [128,40]
print('y:',y.shape)

for k in range(weight.shape[0]):  # 40
    buff = np.dot(x1, weight[k])  # [128,20] * [20,30] => [128,30]
    buff = buff * x2  # [128,30] * [128,30]
    buff = np.sum(buff,axis=1)  # [128,]
    y[:,k] = buff
y += bias
dif = y - arr_output
