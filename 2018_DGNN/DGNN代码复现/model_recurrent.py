import torch
import torch.nn as nn
from torch.nn import init
from combiner import Combiner
from edge_updater import Edge_updater_nn
from node_updater import TLSTM
from scipy.sparse import lil_matrix, find
import numpy as np
from numpy.random import choice
from decayer import Decayer
from attention import Attention
import time

class DyGNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dims, edge_output_size, device, w , is_att = False,transfer=False , nor =0, if_no_time=0, threhold = None, second_order=False, if_updated = 0, drop_p = 0, num_negative = 5 , act = 'tanh', if_propagation = 1 ,decay_method='exp', weight = None, relation_size=None,bias = True):
        super(DyGNN,self).__init__()
        self.embedding_dims = embedding_dims  # 64
        self.num_embeddings = num_embeddings  # 1899,节点数量
        self.nor = nor
        #self.weight = weight.to(device)
        self.device = device
        self.transfer = transfer  # 1
        self.if_propagation = if_propagation  # 1
        self.if_no_time = if_no_time
        self.second_order = second_order
        # self.cuda = cuda
        self.combiner = Combiner(embedding_dims, embedding_dims, act).to(device)  # [64, 64]; 3. merge unit
        self.decay_method = decay_method  # g(delat t)的衰减方式log
        self.if_updated = if_updated
        self.threhold = threhold
        print('Only propagate to relevance nodes below time interval: ', threhold)
        # self.tanh = nn.Tanh().to(device)
        if act == 'tanh':
            self.act = nn.Tanh().to(device)
        elif act == 'sigmoid':
            self.act = nn.Sigmoid().to(device)
        else:
            self.act = nn.ReLU().to(device) 

        self.decayer = Decayer(device, w, decay_method)  # 定义g(delta t)函数
        # 1. interact unit.
        self.edge_updater_head = Edge_updater_nn(embedding_dims, edge_output_size,act, relation_size).to(device)  # interact unit.
        self.edge_updater_tail = Edge_updater_nn(embedding_dims, edge_output_size,act, relation_size).to(device)  # interact unit.
        # 2. update unit.
        if if_no_time:
            self.node_updater_head = nn.LSTMCell(edge_output_size, embedding_dims, bias).to(device)
            self.node_updater_tail = nn.LSTMCell(edge_output_size, embedding_dims, bias).to(device) 
        else:
            self.node_updater_head = TLSTM(edge_output_size, embedding_dims).to(device)  # S-update
            self.node_updater_tail = TLSTM(edge_output_size, embedding_dims).to(device)	 # G-update
        # prop unit. Ws*e(t)
        self.tran_head_edge_head = nn.Linear(edge_output_size, embedding_dims, bias).to(device)
        self.tran_head_edge_tail = nn.Linear(edge_output_size, embedding_dims, bias).to(device)	

        self.tran_tail_edge_head = nn.Linear(edge_output_size, embedding_dims, bias).to(device)  # Ws*e(t)
        self.tran_tail_edge_tail = nn.Linear(edge_output_size, embedding_dims, bias).to(device)
        self.is_att = is_att
        if self.is_att:  # fa
            self.attention = Attention(embedding_dims).to(device)

        self.num_negative = num_negative  # 负采样

        self.recent_timestamp = torch.zeros((num_embeddings, 1), dtype = torch.float, requires_grad = False).to(device)  # [1899, 1]


        self.interaction_timestamp = lil_matrix((num_embeddings,num_embeddings),dtype = np.float32)  # [1899, 1899]; 全0稀疏矩阵
        

        # 节点cell特征
        self.cell_head = nn.Embedding(num_embeddings, embedding_dims, weight).to(device)  # 节点对应到的embedding;
        self.cell_head.weight.requires_grad = False  # 屏蔽权重更新
        self.cell_tail = nn.Embedding(num_embeddings, embedding_dims, weight).to(device)
        self.cell_tail.weight.requires_grad = False
        # 节点hidden特征
        self.hidden_head = nn.Embedding(num_embeddings, embedding_dims, weight).to(device)
        self.hidden_head.weight.requires_grad = False
        self.hidden_tail = nn.Embedding(num_embeddings, embedding_dims, weight).to(device)
        self.hidden_tail.weight.requires_grad = False		
        # 节点特征
        self.node_representations = nn.Embedding(num_embeddings, embedding_dims, weight).to(device)  # 初始化的节点embedding
        self.node_representations.weight.requires_grad = False

        if transfer:
            self.transfer2head = nn.Linear(embedding_dims, embedding_dims, False).to(device)  # 计算loss，Ps
            self.transfer2tail = nn.Linear(embedding_dims, embedding_dims, False).to(device)
            if drop_p>=0:
                self.dropout = nn.Dropout(p=drop_p).to(device)


        self.cell_head_copy = nn.Embedding.from_pretrained(self.cell_head.weight.clone()).to(device)
        self.cell_tail_copy = nn.Embedding.from_pretrained(self.cell_tail.weight.clone()).to(device)
        self.hidden_head_copy = nn.Embedding.from_pretrained(self.hidden_head.weight.clone()).to(device)
        self.hidden_tail_copy = nn.Embedding.from_pretrained(self.hidden_tail.weight.clone()).to(device)
        self.node_representations_copy = nn.Embedding.from_pretrained(self.node_representations.weight.clone()).to(device)

        # if cuda:
        #     self.cell_head = self.cell_head.cuda()
        #     self.cell_tail = self.cell_tail.cuda()
        #     self.node_representations = self.node_representations.cuda()
        #     self.recent_timestamp = self.recent_timestamp.cuda()
        #     self.tran_head_edge_head.cuda()
        #     self.tran_head_edge_head.cuda()
        #     self.tran_tail_edge_head.cuda()
        #     self.tran_tail_edge_tail.cuda()

    def reset_time(self):  # 最近的时间; 交互的时间;
        self.recent_timestamp = torch.zeros((self.num_embeddings, 1), dtype = torch.float, requires_grad = False).to(self.device)
        self.interaction_timestamp = lil_matrix((self.num_embeddings,self.num_embeddings),dtype = np.float32)

    def reset_reps(self):
        self.cell_head = nn.Embedding.from_pretrained(self.cell_head_copy.weight.clone()).to(self.device)
        self.cell_tail = nn.Embedding.from_pretrained(self.cell_tail_copy.weight.clone()).to(self.device)
        self.hidden_head = nn.Embedding.from_pretrained(self.hidden_head_copy.weight.clone()).to(self.device)
        self.hidden_tail = nn.Embedding.from_pretrained(self.hidden_tail_copy.weight.clone()).to(self.device)
        self.node_representations = nn.Embedding.from_pretrained(self.node_representations_copy.weight.clone()).to(self.device)

    def link_pred_with_update(self,test_data):
        pass



    def forward(self,interactions):

        test_time = False

        all_head_nodes = set()
        all_tail_nodes = set()
               
        steps = len(interactions[:,0])  # batch

        node2timetsamp = dict()

        node2cell_head = dict()
        node2cell_tail = dict()
        node2hidden_head = dict()
        node2hidden_tail = dict()
        
        node2rep = dict()

        output_rep_head = []
        output_rep_tail = []
        tail_neg_list = []
        head_neg_list = []

        if test_time:
            old_time = time.time()
        for i in range(steps):
            i_condi = i%200 == 1
            if test_time and i_condi:
                time1 = time.time()
                print('----------------------------------------------------')
                print(i,'1 step time', str(time1 - old_time))
                old_time = time1


            head_index = int(interactions[i,0])  # head节点
            tail_index = int(interactions[i,1])  # tail节点
            all_head_nodes.add(head_index)  # src节点
            all_tail_nodes.add(tail_index)  # dst节点

            head_inx_lt = torch.LongTensor([head_index]).to(self.device)

            tail_inx_lt = torch.LongTensor([tail_index]).to(self.device)


            timestamp = interactions[i,2]  # 时间
            current_t = torch.FloatTensor([timestamp]).view(-1,1).to(self.device)  # [1, 1]
            # self.recent_timestamp: 表示节点最近的一次交互发生时间
            head_prev_t = self.recent_timestamp[head_index]  # self.recent_timestamp:全0节点[1899, 1]; 该节点最近的时间发生的事件;

            tail_prev_t = self.recent_timestamp[tail_index]  #


            if test_time and i_condi:
                time2 = time.time()
                print('test_point2', str(time2-time1))
            # 取节点特征
            if head_index in node2rep:  # 如果节点在node2rep
                head_node_rep = node2rep[head_index]
            else:
                head_node_rep = self.node_representations(head_inx_lt)  # 取初始化head节点的embedding

            if tail_index in node2rep:
                tail_node_rep = node2rep[tail_index]
            else:
                tail_node_rep = self.node_representations(tail_inx_lt)  # 初始化tail的embedding

            # 取head节点的特征(cell, hidden)
            if head_index in node2hidden_head:  #
                head_node_cell_head = node2cell_head[head_index]
                head_node_hidden_head = node2hidden_head[head_index]
            else:
                head_node_cell_head = self.cell_head(head_inx_lt)  # head_cell作为head的embedding
                head_node_hidden_head = self.hidden_head(head_inx_lt)  # hidden_head作为head的embedding
            if head_index in node2hidden_tail:
                head_node_hidden_tail = node2hidden_tail[head_index]
            else:
                head_node_hidden_tail = self.hidden_tail(head_inx_lt)  # hidden_head作为tail的embedding

            # 取tail节点的特征(cell, hidden)
            if tail_index in node2hidden_tail:
                tail_node_cell_tail = node2cell_tail[tail_index]
                tail_node_hidden_tail = node2hidden_tail[tail_index]
            else:
                tail_node_cell_tail = self.cell_tail(tail_inx_lt)  # cell_tail的embedding
                tail_node_hidden_tail = self.hidden_tail(tail_inx_lt)  # hidden_tail的embedding

            if tail_index in node2hidden_head:
                tail_node_hidden_head = node2hidden_head[tail_index]
            else:
                tail_node_hidden_head = self.hidden_head(tail_inx_lt)  # hidden_head的embedding


            if test_time and i_condi:
                time3 = time.time()
                print('prepare rep time', str(time3-time2))

            head_delta_t = current_t - head_prev_t  # 时间差，计算detla Time; 本次发生时间-上次发生时间
            tail_delta_t = current_t - tail_prev_t

            with torch.no_grad():  # 时间不更新梯度
                self.recent_timestamp[[head_index, tail_index]] = current_t  # 将head和tail节点的时间更新;

            transed_head_delta_t = self.decayer(head_delta_t)  # 时间差函数g(delta(t))
            transed_tail_delta_t = self.decayer(tail_delta_t)
            # update component
            # 1. The interact unit.
            edge_info_head = self.edge_updater_head(head_node_rep, tail_node_rep)  # e(t)
            edge_info_tail = self.edge_updater_tail(head_node_rep, tail_node_rep)
            

            # 2. update unit. Head
            if self.if_no_time:
                updated_head_node_hidden_head,updated_head_node_cell_head  = self.node_updater_head(edge_info_head, ( head_node_hidden_head, head_node_cell_head ))
            else:
                updated_head_node_cell_head, updated_head_node_hidden_head = self.node_updater_head(edge_info_head, head_node_cell_head, head_node_hidden_head , transed_head_delta_t)
            # 3. merge unit. Head
            updated_head_node_rep = self.combiner(updated_head_node_hidden_head, head_node_hidden_tail)
            # 更新节点的embedding
            node2cell_head[head_index] = updated_head_node_cell_head
            node2hidden_head[head_index] = updated_head_node_hidden_head
            node2rep[head_index] = updated_head_node_rep


            if self.if_updated:
                output_rep_head.append(updated_head_node_rep)
            else:
                output_rep_head.append(head_node_rep)  # 存储节点embedding
            # Tail的The update component.
            if self.if_no_time:
                updated_tail_node_hidden_tail, updated_tail_node_cell_tail, = self.node_updater_tail(edge_info_tail, (tail_node_hidden_tail, tail_node_cell_tail))
            else:  # update unit
                updated_tail_node_cell_tail, updated_tail_node_hidden_tail = self.node_updater_tail(edge_info_tail, tail_node_cell_tail, tail_node_hidden_tail, transed_tail_delta_t)
            updated_tail_node_rep = self.combiner(tail_node_hidden_head, updated_tail_node_hidden_tail)  # merge unit

            node2cell_tail[tail_index] = updated_tail_node_cell_tail
            node2hidden_tail[tail_index] = updated_tail_node_hidden_tail
            node2rep[tail_index] = updated_tail_node_rep


            if self.if_updated:
                output_rep_tail.append(updated_tail_node_rep)
            else:
                output_rep_tail.append(tail_node_rep)  # 存储tail接待你特征

            if test_time and i_condi:
                time4 = time.time()
                print('update reps', str(time4-time3))

            # The propagation component
            if self.if_propagation:
                head_node_head_neighbors, head_node_tail_neighbors = self.propagation(head_index, current_t, edge_info_head, 'head', node2cell_head, node2hidden_head, node2cell_tail, node2hidden_tail, node2rep, self.threhold, self.second_order)
                tail_node_head_neighbors, tail_node_tail_neighbors = self.propagation(tail_index, current_t, edge_info_tail, 'tail', node2cell_head, node2hidden_head, node2cell_tail, node2hidden_tail, node2rep, self.threhold, self.second_order)
            else:
                head_node_head_neighbors, head_node_tail_neighbors, n_i_1, n_i_2 = self.get_neighbors(head_index,current_t, self.threhold)
                tail_node_head_neighbors, tail_node_tail_neighbors, n_i_1, n_i_2 = self.get_neighbors(tail_index, current_t, self.threhold)
                head_node_head_neighbors = set(head_node_head_neighbors)
                head_node_tail_neighbors = set(head_node_tail_neighbors)
                tail_node_head_neighbors = set(tail_node_head_neighbors)
                tail_node_tail_neighbors = set(tail_node_tail_neighbors)

            if test_time and i_condi:
                time5 = time.time()
                if self.if_propagation:
                    print('propagation time', str(time5-time4))
                else:
                    print('Get neighbors time', str(time5-time4))
            all_head_nodes = all_head_nodes | head_node_head_neighbors | tail_node_head_neighbors
            all_tail_nodes = all_tail_nodes | head_node_tail_neighbors | tail_node_tail_neighbors

            ### generate negative samples ###
            tail_candidates = all_tail_nodes - {head_index,tail_index} - head_node_tail_neighbors  # tail的候选
            if len(tail_candidates)==0:
                tail_neg_samples = list(choice(range(self.num_embeddings), size=self.num_negative))  # 在所有节点中随机采样

            else:
                tail_neg_samples = list(choice(list(tail_candidates), size = self.num_negative))  # 在候选节点中采样

            head_candidates = all_head_nodes - {tail_index,head_index}- tail_node_head_neighbors
            if len(head_candidates) ==0:
                head_neg_samples = list(choice(range(self.num_embeddings), size=self.num_negative))
            else:
                head_neg_samples = list(choice(list(head_candidates), size = self.num_negative))



            if test_time and i_condi: 
                time6 = time.time()
                print('get negative samples time', str(time6 - time5))

            for i in tail_neg_samples:
                if i in node2rep:
                    tail_neg_list.append(node2rep[i])  # 获取节点最新的embedding
                else:
                    i_lt = torch.LongTensor([i]).to(self.device)

                    tail_neg_list.append(self.node_representations(i_lt))
            for i in head_neg_samples:
                if i in node2rep:
                    head_neg_list.append(node2rep[i])
                else:
                    i_lt = torch.LongTensor([i]).to(self.device)

                    head_neg_list.append(self.node_representations(i_lt))
            if test_time and i_condi: 
                time7 = time.time()
                print('Prepare neg reps time', str(time7 - time6))




        ### update interaction time ###
            self.interaction_timestamp[head_index, tail_index] = current_t[0,0]  # 更新当前的时间节点相互作用时间

        ###### Prepare modifed cell, hidden and rep to write back to the memory ########
        cell_head_inx = list(node2cell_head.keys())
        output_cell_head = list(node2cell_head.values())

        cell_tail_inx = list(node2cell_tail.keys())
        output_cell_tail = list(node2cell_tail.values())


        hidden_head_inx = list(node2hidden_head.keys())
        output_hidden_head = list(node2hidden_head.values())

        hidden_tail_inx = list(node2hidden_tail.keys())
        output_hidden_tail = list(node2hidden_tail.values())

        rep_inx = list(node2rep.keys())
        output_rep = list(node2rep.values())


        # head节点特征;
        output_cell_head_tensor = torch.cat([*output_cell_head]).view(-1,self.embedding_dims)  # [5, 64]
        output_hidden_head_tensor = torch.cat([*output_hidden_head]).view(-1,self.embedding_dims)
        output_rep_head_tensor = torch.cat([*output_rep_head]).view(-1,self.embedding_dims)
        # tail节点特征;
        output_cell_tail_tensor = torch.cat([*output_cell_tail]).view(-1,self.embedding_dims)
        output_hidden_tail_tensor = torch.cat([*output_hidden_tail]).view(-1,self.embedding_dims)
        output_rep_tail_tensor = torch.cat([*output_rep_tail]).view(-1,self.embedding_dims)
        # 节点特征;
        output_rep_tensor = torch.cat([*output_rep]).view(-1,self.embedding_dims)
        # 负采样节点; 每个样本负采样5个节点
        tail_neg_tensors = torch.cat([*tail_neg_list]).view(-1,self.embedding_dims)  # [25, 64]
        head_neg_tensors = torch.cat([*head_neg_list]).view(-1,self.embedding_dims)

        if self.transfer:  # 特征投影
            output_rep_head_tensor = self.dropout(self.transfer2head(output_rep_head_tensor))  # pos样本head_tensor [5, 64]
            output_rep_tail_tensor = self.dropout(self.transfer2tail(output_rep_tail_tensor))  # pos样本tail_tensor

            head_neg_tensors =self.dropout(self.transfer2head(head_neg_tensors))  # neg样本head [25, 64]
            tail_neg_tensors = self.dropout(self.transfer2tail(tail_neg_tensors))  # neg样本tail

        if self.nor:  # normalize
            output_rep_head_tensor = nn.functional.normalize(output_rep_head_tensor)
            output_rep_tail_tensor = nn.functional.normalize(output_rep_tail_tensor)

            head_neg_tensors = nn.functional.normalize(head_neg_tensors)
            tail_neg_tensors = nn.functional.normalize(tail_neg_tensors)



        with torch.no_grad():  # 不更新节点embedding
            self.cell_head.weight[cell_head_inx,:] = output_cell_head_tensor
            self.hidden_head.weight[hidden_head_inx,:] = output_hidden_head_tensor

            self.cell_tail.weight[cell_tail_inx,:] = output_cell_tail_tensor
            self.hidden_tail.weight[hidden_tail_inx,:] = output_hidden_tail_tensor

            self.node_representations.weight[rep_inx,:] = output_rep_tensor




        # pos_head, pos_tail,  neg_head, neg_tail;
        return output_rep_head_tensor, output_rep_tail_tensor, head_neg_tensors, tail_neg_tensors



    

    def get_rep(self, nodes, rep_type, rep_dict):
        if rep_type == 'node_rep':
            rep = self.node_representations(torch.LongTensor(nodes).to(self.device))
        elif rep_type == 'cell_head':  # Embedding(1899, 64)
            rep = self.cell_head(torch.LongTensor(nodes).to(self.device))  # neighbor节点的embedding
        elif rep_type == 'cell_tail':
            rep = self.cell_tail(torch.LongTensor(nodes).to(self.device))
        elif rep_type == 'hidden_head':
            rep = self.hidden_head(torch.LongTensor(nodes).to(self.device))    
        else:
            rep = self.hidden_tail(torch.LongTensor(nodes).to(self.device))     
        for nei in nodes:
            if nei in rep_dict:  # 如果在rep_dict中存在，则用这个值; 在之前更新过的embedding中
                rep[nodes.index(nei),:] = rep_dict[nei]  # 如果有节点，则按照rep_dict中的特征
        return rep




    def get_neighbors(self,node,current_t,threhold=None):
        row_inx, col_inx, timestamps = find(self.interaction_timestamp)  # 返回稀疏矩阵A中的非零元的位置以及数值; 节点发生过事件

        # node是tail，找出head节点
        head_inx = list(np.where(col_inx == node)[0])  # 返回位置索引
        head_neighbors = row_inx[head_inx]  # 对应到邻居节点
        head_timestamps = timestamps[head_inx]


        # node是head，找到tail节点
        tail_inx = list(np.where(row_inx == node)[0])
        tail_neighbors = col_inx[tail_inx]
        tail_timestamps = timestamps[tail_inx]
        if threhold is not None:  # 满足时间的τ

            head_inx_th = (current_t.item() -  head_timestamps ) <=threhold
            head_neighbors = head_neighbors[head_inx_th]
            head_timestamps = head_timestamps[head_inx_th]


            tail_inx_th = (current_t.item() - tail_timestamps) <=threhold
            tail_timestamps = tail_timestamps[tail_inx_th]
            tail_neighbors = tail_neighbors[tail_inx_th]







        return head_neighbors, tail_neighbors , head_timestamps, tail_timestamps




    def get_att_score(self,node, neighbors, node2rep):
        nei_reps = self.get_rep(neighbors, 'node_rep', node2rep)  # neighbor的特征
        node_rep  = self.get_rep([node], 'node_rep', node2rep)  # node的特征

        node_reps = node_rep.repeat(len(neighbors),1)  # node节点重复n次，和neighbors数量相同

        return self.attention(node_reps, nei_reps)

        

    def propagation(self, node, current_t, edge_info, node_type, node2cell_head, node2hidden_head, node2cell_tail, node2hidden_tail, node2rep, threhold = None, second_order=False):


        # Get neighbors; 获得node的head和tail的邻居节点;
        head_neighbors, tail_neighbors, head_timestamps, tail_timestamps = self.get_neighbors(node, current_t,threhold)


        head_neighbors = list(head_neighbors)
        head_timestamps = list(head_timestamps)
        if len(head_neighbors)>0:
            if node_type == 'head':  # 6节点的类型
                head_nei_edge_info = self.tran_head_edge_head(edge_info)

            else:
                head_nei_edge_info = self.tran_tail_edge_head(edge_info)  # edge_info=e(t)
            # 计算时间差; 当前时间 - 邻居节点最近出现的时间;
            head_delta_ts = current_t.repeat(len(head_timestamps),1) - torch.FloatTensor(head_timestamps).to(self.device).view(-1,1)
            transed_head_delta_ts = self.decayer(head_delta_ts)  # g(delta t)



            # heihbors的cell更新
            head_nei_cell = self.get_rep(head_neighbors, 'cell_head',node2cell_head)  # 获取节点embedding
            if self.if_no_time:  # 不使用delta t
                tran_head_nei_edge_info = head_nei_edge_info.repeat(len(head_neighbors),1)
            else:
                tran_head_nei_edge_info = head_nei_edge_info.repeat(len(head_neighbors),1) * transed_head_delta_ts


            if self.is_att:  # 计算fa()
                att_score_head = self.get_att_score(node, head_neighbors, node2rep)  # attention系数
                tran_head_nei_edge_info = tran_head_nei_edge_info*att_score_head  # 更新节点特征fa


            head_nei_cell = head_nei_cell + tran_head_nei_edge_info  # 更新总的节点特征
            head_nei_hidden = self.act(head_nei_cell)
            # 3. merge unit.
            head_nei_tail_hidden = self.get_rep(head_neighbors, 'hidden_tail', node2hidden_tail)
            head_nei_rep = self.combiner(head_nei_hidden, head_nei_tail_hidden)
            # 更新节点embedding
            for i, nei in enumerate(head_neighbors):
                node2cell_head[nei] = head_nei_cell[i].view(-1,self.embedding_dims)
                node2hidden_head[nei] = head_nei_hidden[i].view(-1,self.embedding_dims)
                node2rep[nei] = head_nei_rep[i].view(-1,self.embedding_dims)

            if second_order:  # 节点的二阶邻居更新;  邻居的邻居节点
                for  head_node_sec in head_neighbors:
                    self.second_propagation(head_node_sec, current_t , tran_head_nei_edge_info[0,:], 'head', node2cell_head, node2hidden_head, node2cell_tail, node2hidden_tail, node2rep, threhold)







        tail_neighbors = list(tail_neighbors)
        tail_timestamps = list(tail_timestamps)
        if len(tail_neighbors)>0:

            if node_type == 'head':
                tail_nei_edge_info = self.tran_head_edge_tail(edge_info)
            else: 
                tail_nei_edge_info = self.tran_tail_edge_tail(edge_info)

            tail_delta_ts = current_t.repeat(len(tail_timestamps),1) - torch.FloatTensor(tail_timestamps).to(self.device).view(-1,1) 
            transed_tail_delta_ts = self.decayer(tail_delta_ts)


            tail_nei_cell = self.get_rep(tail_neighbors, 'cell_tail', node2cell_tail)
            if self.if_no_time:
                tran_tail_nei_edge_info = tail_nei_edge_info.repeat(len(tail_neighbors),1)
            else:
                tran_tail_nei_edge_info = tail_nei_edge_info.repeat(len(tail_neighbors),1) * transed_tail_delta_ts

            if self.is_att:
                att_score_tail = self.get_att_score(node, tail_neighbors, node2rep)
                tran_head_nei_edge_info = tran_tail_nei_edge_info*att_score_tail


            tail_nei_cell = tail_nei_cell + tran_tail_nei_edge_info


            tail_nei_hidden = self.act(tail_nei_cell)


            tail_nei_head_hidden = self.get_rep(tail_neighbors, 'hidden_head', node2hidden_head)

            tail_nei_rep = self.combiner(tail_nei_head_hidden, tail_nei_hidden)


            for i, nei in enumerate(tail_neighbors):
                node2cell_tail[nei] = tail_nei_cell[i].view(-1,self.embedding_dims)
                node2hidden_tail[nei] = tail_nei_hidden[i].view(-1,self.embedding_dims)
                node2rep[nei]= tail_nei_rep[i].view(-1,self.embedding_dims)

            if second_order:
                for  tail_node_sec in tail_neighbors:
                    self.second_propagation(tail_node_sec, current_t , tran_tail_nei_edge_info[0,:], 'tail', node2cell_head, node2hidden_head, node2cell_tail, node2hidden_tail, node2rep, threhold)

        return set(head_neighbors), set(tail_neighbors)


    def second_propagation(self, node, current_t, edge_info, node_type, node2cell_head, node2hidden_head, node2cell_tail, node2hidden_tail, node2rep, threhold = None):

        head_neighbors, tail_neighbors, head_timestamps, tail_timestamps = self.get_neighbors(node,current_t, threhold)


        head_neighbors = list(head_neighbors)
        head_timestamps = list(head_timestamps)
        if len(head_neighbors)>0:
            if node_type == 'head':
                head_nei_edge_info = self.tran_head_edge_head(edge_info)

            else: 
                head_nei_edge_info = self.tran_tail_edge_head(edge_info)

            head_delta_ts = current_t.repeat(len(head_timestamps),1) - torch.FloatTensor(head_timestamps).to(self.device).view(-1,1)
            transed_head_delta_ts = self.decayer(head_delta_ts)




            head_nei_cell = self.get_rep(head_neighbors, 'cell_head',node2cell_head)
            if self.if_no_time:
                tran_head_nei_edge_info = head_nei_edge_info.repeat(len(head_neighbors),1)
            else:
                tran_head_nei_edge_info = head_nei_edge_info.repeat(len(head_neighbors),1) * transed_head_delta_ts


            if self.is_att:
                att_score_head = self.get_att_score(node, head_neighbors, node2rep)
                tran_head_nei_edge_info = tran_head_nei_edge_info*att_score_head


            head_nei_cell = head_nei_cell + tran_head_nei_edge_info
            head_nei_hidden = self.act(head_nei_cell)
            head_nei_tail_hidden = self.get_rep(head_neighbors, 'hidden_tail', node2hidden_tail)
            head_nei_rep = self.combiner(head_nei_hidden, head_nei_tail_hidden)

            for i, nei in enumerate(head_neighbors):
                node2cell_head[nei] = head_nei_cell[i].view(-1,self.embedding_dims)
                node2hidden_head[nei] = head_nei_hidden[i].view(-1,self.embedding_dims)
                node2rep[nei] = head_nei_rep[i].view(-1,self.embedding_dims)


        

        tail_neighbors = list(tail_neighbors)
        tail_timestamps = list(tail_timestamps)
        if len(tail_neighbors)>0:

            if node_type == 'head':
                tail_nei_edge_info = self.tran_head_edge_tail(edge_info)
            else: 
                tail_nei_edge_info = self.tran_tail_edge_tail(edge_info)

            tail_delta_ts = current_t.repeat(len(tail_timestamps),1) - torch.FloatTensor(tail_timestamps).to(self.device).view(-1,1) 
            transed_tail_delta_ts = self.decayer(tail_delta_ts)



            tail_nei_cell = self.get_rep(tail_neighbors, 'cell_tail', node2cell_tail)
            if self.if_no_time:
                tran_tail_nei_edge_info = tail_nei_edge_info.repeat(len(tail_neighbors),1)
            else:
                tran_tail_nei_edge_info = tail_nei_edge_info.repeat(len(tail_neighbors),1) * transed_tail_delta_ts

            if self.is_att:
                att_score_tail = self.get_att_score(node, tail_neighbors, node2rep)
                tran_head_nei_edge_info = tran_tail_nei_edge_info*att_score_tail


            tail_nei_cell = tail_nei_cell + tran_tail_nei_edge_info


            tail_nei_hidden = self.act(tail_nei_cell)


            tail_nei_head_hidden = self.get_rep(tail_neighbors, 'hidden_head', node2hidden_head)

            tail_nei_rep = self.combiner(tail_nei_head_hidden, tail_nei_hidden)


            for i, nei in enumerate(tail_neighbors):
                node2cell_tail[nei] = tail_nei_cell[i].view(-1,self.embedding_dims)
                node2hidden_tail[nei] = tail_nei_hidden[i].view(-1,self.embedding_dims)
                node2rep[nei]= tail_nei_rep[i].view(-1,self.embedding_dims)
        return set(head_neighbors), set(tail_neighbors)

    def loss(self, interactions):
        # 前向计算
        output_rep_head_tensor, output_rep_tail_tensor, head_neg_tensors, tail_neg_tensors = self.forward(interactions)
        # 正样本复制n次，保持和负采样一致; [5, 64] ->[25, 64]
        head_pos_tensors = output_rep_head_tensor.clone().repeat(1,self.num_negative).view(-1,self.embedding_dims)
        tail_pos_tensors = output_rep_tail_tensor.clone().repeat(1,self.num_negative).view(-1,self.embedding_dims)

        num_pp = output_rep_head_tensor.size()[0]
        labels_p = torch.FloatTensor([1]*num_pp).to(self.device)  # 正样本数量
        labels_n = torch.FloatTensor([0]*num_pp*2*self.num_negative).to(self.device)  # 负样本数量

        labels = torch.cat((labels_p,labels_n))

                            # [5, 1, 64]                                                # [5, 64, 1]
        scores_p = torch.bmm(output_rep_head_tensor.view(num_pp,1,self.embedding_dims),output_rep_tail_tensor.view(num_pp,self.embedding_dims,1))
        scores_n_1 = torch.bmm(head_neg_tensors.view(num_pp*self.num_negative,1,self.embedding_dims), tail_pos_tensors.view(num_pp*self.num_negative, self.embedding_dims,1))
        scores_n_2 = torch.bmm(head_pos_tensors.view(num_pp*self.num_negative,1,self.embedding_dims), tail_neg_tensors.view(num_pp*self.num_negative, self.embedding_dims,1))

        scores = torch.cat((scores_p,scores_n_1,scores_n_2)).view(num_pp*(1+2*self.num_negative))  # 结果拼接



        bce_with_logits_loss = nn.BCEWithLogitsLoss()
        loss = bce_with_logits_loss(scores,labels)

        return loss










        # for i,nei in enumerate(head_neighbors):
        #     node2cell_head[nei] = head_nei_cell[i]
        #     node2hidden_head[nei] = self.tanh(head_nei_cell[i])
        #     if nei in node2hidden_tail:
        #         node2rep[nei] = self.combiner(node2hidden_head[nei], node2hidden_tail[nei])
        #     else:
        #         node2rep[nei] = self.combiner(node2hidden_head[nei], self.hidden_tail(torch.LongTensor([nei])))

        # for i, nei in enumerate(tail_neighbors):
        #     node2cell_tail[nei] = tail_nei_cell[i]
        #     node2hidden_tail[nei] = self.tanh(tail_nei_cell[i])
        #     if nei in node2hidden_head:
        #         node2rep[nei] = self.combiner(node2hidden_head[nei], node2hidden_tail[nei])
        #     else:
        #         node2rep[nei] = self.combiner(self.hidden_head(torch.LongTensor([nei])), node2hidden_tail)
        # for ind, nei in head_neighbors:
        #     if nei in node2cell_head:
        #         head_nei_cell = node2cell_head[nei]
        #     else:
        #         head_nei_cell = self.cell_head(torch.LongTensor([node]))










            





