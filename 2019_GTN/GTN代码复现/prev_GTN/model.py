import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt
import pdb


class GTN(nn.Module): # GTN模型实现
    
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class, num_layers=2, norm):
        super(GTN, self).__init__()
        self.num_edge = num_edge # 边的类别数
        self.num_channels = num_channels # 通道的数目
        self.w_in = w_in # 输入特征
        self.w_out = w_out # 输出特征
        self.num_class = num_class # 分类的类别数
        self.num_layers = num_layers # 层数
        self.is_norm = norm # 是否标准化
        layers = []

        for i in range(num_layers): # 卷积层数
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, first=True)) # 第一层
            else:
                layers.append(GTLayer(num_edge, num_channels, first=False)) # 后续层

        self.layers = nn.ModuleList(layers)

        self.weight = nn.Parameter(torch.Tensor(w_in, w_out)) # 权重参数
        self.bias = nn.Parameter(torch.Tensor(w_out)) # 偏置参数
        self.loss = nn.CrossEntropyLoss() # 损失

        self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out) # 线性层1

        self.linear2 = nn.Linear(self.w_out, self.num_class) # 线性层2
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def gcn_conv(self,X,H): # GCN操作
        X = torch.mm(X, self.weight)
        H = self.norm(H, add=True)
        return torch.mm(H.t(),X)

    def normalization(self, H): # 归一化
        for i in range(self.num_channels):
            if i==0:
                H_ = self.norm(H[i,:,:]).unsqueeze(0)
            else:
                H_ = torch.cat((H_,self.norm(H[i,:,:]).unsqueeze(0)), dim=0)
        return H_

    def norm(self, H, add=False): # 按列进行归一化
        H = H.t()
        if add == False:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor)) # 去除自连接的结果
        else:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor)) + torch.eye(H.shape[0]).type(torch.FloatTensor)
        deg = torch.sum(H, dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv*torch.eye(H.shape[0]).type(torch.FloatTensor)
        H = torch.mm(deg_inv,H)
        H = H.t()
        return H

    def forward(self, A, X, target_x, target):
        A = A.unsqueeze(0).permute(0,3,1,2) # A.shape=(1,5,18405,18405)
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                H = self.normalization(H)
                H, W = self.layers[i](A, H) 
            Ws.append(W)
        '''
        H = A(l)
        '''
        #H,W1 = self.layer1(A)
        #H = self.normalization(H)
        #H,W2 = self.layer2(A, H)
        #H = self.normalization(H)
        #H,W3 = self.layer3(A, H)

        for i in range(self.num_channels): # 对每个通道单独进行GCN变换
            if i==0:
                X_ = F.relu(self.gcn_conv(X,H[i]))
            else:
                X_tmp = F.relu(self.gcn_conv(X,H[i]))
                X_ = torch.cat((X_,X_tmp), dim=1)
        
        # 进行两次线性层变换，得到最终的分类结果
        X_ = self.linear1(X_)
        X_ = F.relu(X_)
        y = self.linear2(X_[target_x])

        loss = self.loss(y, target) # 交叉熵损失计算
        return loss, y, Ws

class GTLayer(nn.Module): # GTN单层实现
    
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels # 输入通道数 , in = 5
        self.out_channels = out_channels  # 输出通道数, out = 2
        self.first = first # 是否为第一层
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)
    
    def forward(self, A, H_=None):
        if self.first == True:
            a = self.conv1(A) # GTConv=>[2,N,N] #Q1
            b = self.conv2(A) # Q2
            # 作了第一次矩阵相乘，得到A1
            H = torch.bmm(a,b)  
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            a = self.conv1(A) # 第二层之后只有一个conv1; output:Conv输出归一化edge后的结果
            H = torch.bmm(H_,a) # H_为上一层输出的结果矩阵A1; 输出这一层后的结果为A2
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]

        return H,W # H = A(1)...A(l); W = 归一化后的权重矩阵

class GTConv(nn.Module): # GTN 卷积层实现
    
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,1,1))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        '''
        0) 对weight(conv)进行softmax;
        1)对每个节点在每个edge Type上进行[2,5,1,1]的卷积操作;
        2)对每个edgeType进行加权求和,加权是通过0)softmax。
        '''
        # F.softmax(self.weight，dim=1)对self.weight做softmax:[2，5，1，1]
        # A:[1，5，8994，8994]:带有edgeType的邻接矩阵
        # [1，5，8994，8994]*[2，5，1，1] = [2，5，8994，8994]
        # sum:[2，8994，8994]
        A = torch.sum(A*F.softmax(self.weight, dim=1), dim=1)
        return A
