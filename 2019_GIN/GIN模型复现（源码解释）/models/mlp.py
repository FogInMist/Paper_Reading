import torch
import torch.nn as nn
import torch.nn.functional as F

###MLP with lienar output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model. # 层数，不包括输入层。如果 num_layers 小于 1，则抛出异常。
            input_dim: dimensionality of input features # 输入特征的维度。
            hidden_dim: dimensionality of hidden units at ALL layers # 隐藏层的维度。
            output_dim: number of classes for prediction # 预测的类别数。
            device: which device to use # 用于存储模型的设备。
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim) # 若num_layers == 1，则代表是单层线性神经网络
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList() # 线形层
            self.batch_norms = torch.nn.ModuleList() # 批量标准化
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x): 
        # 表示当前模型为线性模型，直接返回 self.linear(x) 进行预测
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        # 表示当前模型为多层感知器（MLP），需要进行一些操作
        else:
            #If MLP
            h = x

            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h))) # 1.线性层处理 2.批量归一化处理 3. 激活层处理

            return self.linears[self.num_layers - 1](h) # 最后一次线性处理，然后输出结果