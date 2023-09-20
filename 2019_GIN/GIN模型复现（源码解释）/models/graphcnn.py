import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("models/")
from mlp import MLP

class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type, device):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer) # 整数,表示模型的层数,包括输入层和输出层。
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer) # 整数,表示MLP的层数,不包括输入层。
            input_dim: dimensionality of input features # 整数,表示输入特征的维度。
            hidden_dim: dimensionality of hidden units at ALL layers # 整数,表示隐藏层的维度。
            output_dim: number of classes for prediction # 整数,表示输出类的数量。
            final_dropout: dropout ratio on the final linear layer # 浮点数,表示最后层dropout的比率。
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether. # 布尔值,表示是否学习epsilon以区分中心节点和邻居节点。
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max) # 字符串,表示图池化的类型,可以是"max"、"average"或"sum"。
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average) # 字符串,表示邻居池化的类型，可以是"max"、"average"或"sum"。
            device: which device to use # PyTorch设备,表示模型运行的设备。
        '''

        super(GraphCNN, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1))

        ###List of MLPs
        self.mlps = torch.nn.ModuleList() # 多层感知器

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList() # 批量归一化

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function that maps the hidden representation at dofferemt layers into a prediction score
        # 将不同层的隐藏表示映射为预测分数的线性函数
        self.linears_prediction = torch.nn.ModuleList() # 线性预测层

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))


    # 定义了一个用于存储填充后的邻居列表的变量
    def __preprocess_neighbors_maxpool(self, batch_graph):
        ###create padded_neighbor_list in concatenated graph

        #compute the maximum number of neighbors within the graphs in the current minibatch
        max_deg = max([graph.max_neighbor for graph in batch_graph]) # 计算批次数据中的最大邻居数量

        #create a list of padded neighbor lists
        padded_neighbor_list = [] # 用于存储填充后的邻居列表。
        start_idx = [0]  # 用于存储每个图的起始索引。


        #iterate through the graphs in the current minibatch
        for i, graph in enumerate(batch_graph):
            #increment the start index of the current graph
            start_idx.append(start_idx[i] + len(graph.g))
            #create a list of padded neighbors
            padded_neighbors = []
            #iterate through the neighbors of the current graph
            for j in range(len(graph.neighbors)):
                #add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                #padding, dummy data is assumed to be stored in -1
                pad.extend([-1]*(max_deg - len(pad)))

                #Add center nodes in the maxpooling if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
                if not self.learn_eps:
                    pad.append(j + start_idx[i])

                #append the padded neighbor to the list of padded neighbors
                padded_neighbors.append(pad)
            #append the list of padded neighbors to the list of padded neighbor lists
            padded_neighbor_list.extend(padded_neighbors)

        #convert the list of padded neighbor lists to a tensor
        return torch.LongTensor(padded_neighbor_list)


    # 定义了一个用于存储块对角矩阵的变量
    def __preprocess_neighbors_sumavepool(self, batch_graph):
        ###create block diagonal sparse matrix

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        #Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.

        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1],start_idx[-1]]))

        return Adj_block.to(self.device)


    # 定义了一个用于存储graph pooling结果的变量
    def __preprocess_graphpool(self, batch_graph):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)
        
        start_idx = [0]

        #compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling
            if self.graph_pooling_type == "average":
                elem.extend([1./len(graph.g)]*len(graph.g))
            
            else:
            ###sum pooling
                elem.extend([1]*len(graph.g))

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0,1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))
        
        return graph_pool.to(self.device)

    # 最大池化
    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim = 0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim = 1)[0]

        return pooled_rep

    # 图聚合层
    def next_layer_eps(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes separately by epsilon reweighting. 

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                #If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        #Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer])*h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h


    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes altogether  
            
        if self.neighbor_pooling_type == "max": # 求最大值，则是GraphSAGE
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            pooled = torch.spmm(Adj_block, h) # 不求平均， 则是GIN

            if self.neighbor_pooling_type == "average": # 求平均，则是GCN
                #If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        #representation of neighboring and center nodes 
        pooled_rep = self.mlps[layer](pooled)

        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h


    def forward(self, batch_graph):
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        graph_pool = self.__preprocess_graphpool(batch_graph)

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)

        #list of hidden representation at each layer (including input)
        hidden_rep = [X_concat]
        h = X_concat

        for layer in range(self.num_layers-1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list = padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block = Adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_neighbor_list = padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block = Adj_block)

            hidden_rep.append(h)

        score_over_layer = 0
    
        #perform pooling over all nodes in each graph in every layer
        for layer, h in enumerate(hidden_rep):
            pooled_h = torch.spmm(graph_pool, h)
            score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout, training = self.training)

        return score_over_layer  # 最终得分层，类别
