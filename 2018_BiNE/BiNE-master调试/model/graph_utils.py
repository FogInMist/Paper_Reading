__author__ = 'CLH'

import networkx as nx
import graph
import random
from networkx.algorithms import bipartite as bi
import numpy as np
from lsh import get_negs_by_lsh
from io import open
import os
import itertools

class GraphUtils(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.G = nx.Graph()
        self.edge_dict_u = {}
        self.edge_dict_v = {}
        self.edge_list = []
        self.node_u = []
        self.node_v = []
        self.authority_u, self.authority_v = {}, {}
        self.walks_u, self.walks_v = [], []
        self.G_u, self.G_v = None, None
        self.fw_u = os.path.join(self.model_path, "homogeneous_u.dat")
        self.fw_v = os.path.join(self.model_path, "homogeneous_v.dat")
        self.negs_u = {}
        self.negs_v = {}
        self.context_u = {}
        self.context_v = {}

    # 构造训练所需的二部图
    def construct_training_graph(self, filename=None): 
        if filename is None:
            filename = os.path.join(self.model_path, "ratings_train.dat")
        edge_list_u_v = []
        edge_list_v_u = []
        with open(filename, encoding="UTF-8") as fin:
            line = fin.readline()
            while line:
                user, item, rating = line.strip().split("\t")
                if self.edge_dict_u.get(user) is None: # 构造u连接到的节点
                    self.edge_dict_u[user] = {}
                if self.edge_dict_v.get(item) is None: # 构造v连接到的节点
                    self.edge_dict_v[item] = {}
                edge_list_u_v.append((user, item, float(rating))) # 建立边的list(u, v, edge)
                self.edge_dict_u[user][item] = float(rating) # 记录点边关系u -> v
                self.edge_dict_v[item][user] = float(rating) # 记录点边关系v -> u
                edge_list_v_u.append((item, user, float(rating))) # 建立边的list(v, u, edge)
                line = fin.readline()
        # create bipartite graph
        self.node_u = self.edge_dict_u.keys() # u类别的所有节点
        self.node_v = self.edge_dict_v.keys() # v类别的所有节点
        # self.node_u.sort() # 对节点进行排序，python2可用
        # self.node_v.sort()
        self.G.add_nodes_from(self.node_u, bipartite=0) # 给二部图加入u节点
        self.G.add_nodes_from(self.node_v, bipartite=1) # 给二部图加入v节点
        self.G.add_weighted_edges_from(edge_list_u_v+edge_list_v_u) # 将异构图边的信息导入到图中
        self.edge_list = edge_list_u_v # 图的边信息列表

    # 计算节点中心性
    def calculate_centrality(self, mode='hits'): 
        if mode == 'degree_centrality':
            a = nx.degree_centrality(self.G)
        else:
            h, a = nx.hits(self.G) # hub ,authority

        max_a_u, min_a_u,max_a_v,min_a_v = 0, 100000, 0, 100000

        for node in self.G.nodes(): # 求u,v节点中心性的最大和最小值
            if node[0] == "u":
                if max_a_u < a[node]:
                    max_a_u = a[node]
                if min_a_u > a[node]:
                    min_a_u = a[node]
            if node[0] == "i":
                if max_a_v < a[node]:
                    max_a_v = a[node]
                if min_a_v > a[node]:
                    min_a_v = a[node]

        for node in self.G.nodes(): # 计算每一个节点归一化后的中心值大小 authority
            if node[0] == "u":
                if max_a_u-min_a_u != 0:
                    self.authority_u[node] = (float(a[node])-min_a_u) / (max_a_u-min_a_u)
                else:
                    self.authority_u[node] = 0
            if node[0] == 'i':
                if max_a_v-min_a_v != 0:
                    self.authority_v[node] = (float(a[node])-min_a_v) / (max_a_v-min_a_v)
                else:
                    self.authority_v[node] = 0

    # 随机游走采样过程，
    def homogeneous_graph_random_walks(self, percentage, maxT, minT):
        # print(len(self.node_u),len(self.node_v))
        A = bi.biadjacency_matrix(self.G, self.node_u, self.node_v, dtype=np.float,weight='weight', format='csr') # 构造二部图的邻接矩阵
        row_index = dict(zip(self.node_u, itertools.count())) # u类别的节点索引
        col_index = dict(zip(self.node_v, itertools.count())) # v类别的节点索引
        index_row = dict(zip(row_index.values(), row_index.keys())) # 索引对应的u节点
        index_item = dict(zip(col_index.values(), col_index.keys())) # 索引对应的v节点
        AT = A.transpose() # 矩阵转置
        self.save_homogenous_graph_to_file(A.dot(AT),self.fw_u, index_row, index_row) # 保留u节点隐式关系
        self.save_homogenous_graph_to_file(AT.dot(A),self.fw_v, index_item, index_item) # 保留v节点隐式关系
        self.G_u, self.walks_u = self.get_random_walks_restart(self.fw_u, self.authority_u, percentage=percentage, maxT=maxT, minT=minT)
        self.G_v, self.walks_v = self.get_random_walks_restart(self.fw_v, self.authority_v, percentage=percentage, maxT=maxT, minT=minT)

    # 对每个类别（u,v）进行随机游走
    def get_random_walks_restart(self, datafile, hits_dict, percentage, maxT, minT):
        if datafile is None:
            datafile = os.path.join(self.model_path,"rating_train.dat")
        G = graph.load_edgelist(datafile, undirected=True) # 导入图关系

        print("number of nodes: {}".format(len(G.nodes())))
        print("walking...")
        walks = graph.build_deepwalk_corpus_random(G, hits_dict, percentage=percentage, maxT = maxT, minT = minT, alpha=0)
        print("walking...ok")
        return G, walks # 返回图和生成的随机游走序列


    def homogeneous_graph_random_walks_for_large_bipartite_graph(self, percentage, maxT, minT):
        A = bi.biadjacency_matrix(self.G, self.node_u, self.node_v, dtype=np.float,weight='weight', format='csr')
        row_index = dict(zip(self.node_u, itertools.count()))
        col_index = dict(zip(self.node_v, itertools.count()))
        index_row = dict(zip(row_index.values(), row_index.keys()))
        index_item = dict(zip(col_index.values(), col_index.keys()))
        AT = A.transpose()
        matrix_u = self.get_homogenous_graph(A.dot(AT), self.fw_u, index_row, index_row)
        matrix_v = self.get_homogenous_graph(AT.dot(A), self.fw_v, index_item, index_item)
        self.G_u, self.walks_u = self.get_random_walks_restart_for_large_bipartite_graph(matrix_u, self.authority_u, percentage=percentage, maxT=maxT, minT=minT)
        self.G_v, self.walks_v = self.get_random_walks_restart_for_large_bipartite_graph(matrix_v, self.authority_v, percentage=percentage, maxT=maxT, minT=minT)


    def homogeneous_graph_random_walks_for_large_bipartite_graph_without_generating(self, datafile, percentage, maxT, minT):
        self.G_u, self.walks_u = self.get_random_walks_restart_for_large_bipartite_graph_without_generating(datafile, self.authority_u, percentage=percentage, maxT=maxT, minT=minT, node_type='u')
        self.G_v, self.walks_v = self.get_random_walks_restart_for_large_bipartite_graph_without_generating(datafile, self.authority_v, percentage=percentage, maxT=maxT, minT=minT,node_type='i')


    def get_random_walks_restart_for_large_bipartite_graph(self, matrix, hits_dict, percentage, maxT, minT):
        G = graph.load_edgelist_from_matrix(matrix, undirected=True)
        print("number of nodes: {}".format(len(G.nodes())))
        print("walking...")
        walks = graph.build_deepwalk_corpus_random(G, hits_dict, percentage=percentage, maxT = maxT, minT = minT, alpha=0)
        print("walking...ok")
        return G, walks


    def get_random_walks_restart_for_large_bipartite_graph_without_generating(self, datafile, hits_dict, percentage, maxT, minT, node_type='u'):
        if datafile is None:
            datafile = os.path.join(self.model_path,"rating_train.dat")
        G = graph.load_edgelist(datafile, undirected=True)
        cnt = 0
        for n in G.nodes():
            if n[0] == node_type:
                cnt += 1
        print("number of nodes: {}".format(cnt))
        print("walking...")
        walks = graph.build_deepwalk_corpus_random_for_large_bibartite_graph(G, hits_dict, percentage=percentage, maxT = maxT, minT = minT, alpha=0,node_type=node_type)
        # print(walks)
        print("walking...ok")
        return G, walks


    def save_words_and_sentences_to_file(self, filenodes, filesentences):
        with open(filenodes,"w") as fw:
            for node in self.G.keys():
                fw.write(node+"\n")

        with open(filesentences,"w") as fs:
            for nodes in self.walks:
                for index in range(0,len(nodes)):
                    if index == len(nodes)-1:
                        fs.write(nodes[index]+"\n")
                    else:
                        fs.write(nodes[index]+" ")

    # 获取节点的负采样结果                    
    def get_negs(self,num_negs):
        self.negs_u, self.negs_v = get_negs_by_lsh(self.edge_dict_u,self.edge_dict_v,num_negs)
        # print(len(self.negs_u),len(self.negs_v))
        return self.negs_u, self.negs_v

    # 
    def get_context_and_fnegatives(self,G,walks,win_size,num_negs,table):
        # generate context and negatives
        if isinstance(G, graph.Graph):
            node_list = G.nodes()
        elif isinstance(G, list):
            node_list = G
        word2id = {}
        for i in range(len(node_list)):
            word2id[node_list[i]] = i + 1
        walk_list = walks
        print("context...")
        context_dict = {} # 上下文
        new_neg_dict = {} # 负采样
        for step in range(len(walk_list)):
            walk = walk_list[step % len(walk_list)]
            # print(walk)
            batch_labels = []
            # travel each walk
            for iter in range(len(walk)):
                start = max(0, iter - win_size)
                end = min(len(walk), iter + win_size + 1)
                # index: index in window
                if context_dict.get(walk[iter]) is None:
                    context_dict[walk[iter]] = []
                    new_neg_dict[walk[iter]] = []
                labels_list = []
                neg_sample = []
                for index in range(start, end):
                    labels_list.append(walk[index])
                while len(neg_sample) < num_negs:
                    sa = random.choice(range(len(node_list)))
                    if table[sa] in labels_list:
                        continue
                    neg_sample.append(table[sa])
                context_dict[walk[iter]].append(labels_list)
                new_neg_dict[walk[iter]].append(neg_sample)
            if len(batch_labels) == 0:
                continue
        print("context...ok")
        return context_dict, new_neg_dict

    def get_context_and_negatives(self,G,walks,win_size,num_negs,negs_dict):
        # generate context and negatives
        if isinstance(G, graph.Graph):
            node_list = list(G.nodes())
        elif isinstance(G, list):
            node_list = G
        word2id = {}
        for i in range(len(node_list)):
            word2id[node_list[i]] = i + 1
        walk_list = walks
        print("context...")
        context_dict = {}
        new_neg_dict = {}
        for step in range(len(walk_list)):
            walk = walk_list[step % len(walk_list)]
            # print(walk)
            # travel each walk
            for iter in range(len(walk)):
                start = max(0, iter - win_size)
                end = min(len(walk), iter + win_size + 1)
                # index: index in window
                if context_dict.get(walk[iter]) is None:
                    context_dict[walk[iter]] = []
                    new_neg_dict[walk[iter]] = []
                labels_list = []
                negs = negs_dict[walk[iter]]  # 负采样词
                for index in range(start, end):
                    if walk[index] in negs:
                        negs.remove(walk[index]) # 上文词如果在负采样中，则在其负采样中移除
                    if walk[index] == walk[iter]: # 中心词本身，不是上下文词，不做操作
                        continue
                    else:
                        labels_list.append(walk[index]) # 加入到上下文词中
                neg_sample = random.sample(negs,min(num_negs,len(negs))) # 随机负采样四个节点
                context_dict[walk[iter]].append(labels_list) # 上下文词字典
                new_neg_dict[walk[iter]].append(neg_sample) # 负采样词字典
        print("context...ok")
        return context_dict, new_neg_dict # 返回上文词节点和负采样词节点
        # with open(context_file,'w', encoding='utf-8') as fw1, open(neg_file,'w', encoding='utf-8') as fw2:
        #     for u in context_dict.keys():
        #         fw1.write(u+"\t")
        #         fw2.write(u+"\t")
        #         lens = len(context_dict[u])
        #         for i in range(lens):
        #             str1 = u','.join(context_dict[u][i])
        #             str2 = u','.join(neg_dict[u][i])
        #             if i != lens -1:
        #                 fw1.write(str1+"\t")
        #                 fw2.write(str2+"\t")
        #             else:
        #                 fw1.write(str1+"\n")
        #                 fw2.write(str2+"\n")
        # return context_dict, neg_dict

    # 保留计算图结果
    def save_homogenous_graph_to_file(self, A, datafile, index_row, index_item):
        (M,N) = A.shape
        csr_dict = A.__dict__
        data = csr_dict.get("data")
        indptr = csr_dict.get("indptr")
        indices = csr_dict.get("indices")
        col_index = 0
        with open(datafile,'w') as fw:
            for row in range(M):
                for col in range(indptr[row],indptr[row+1]):
                    r = row
                    c = indices[col]
                    fw.write(index_row.get(r)+"\t"+index_item.get(c)+"\t"+str(data[col_index])+"\n")
                    col_index += 1

    def get_homogenous_graph(self, A, datafile, index_row, index_item):
        (M,N) = A.shape
        csr_dict = A.__dict__
        data = csr_dict.get("data")
        indptr = csr_dict.get("indptr")
        indices = csr_dict.get("indices")
        col_index = 0
        matrix = {}
        with open(datafile,'w') as fw:
            for row in range(M):
                for col in range(indptr[row],indptr[row+1]):
                    r = index_row.get(row)
                    c = index_item.get(indices[col])
                    if matrix.get(r) is None:
                        matrix[r] = []
                    matrix[r].append(c)
                    col_index += 1

        return matrix

    def read_sentences_and_homogeneous_graph(self, filesentences=None, datafile=None):
        G = graph.load_edgelist(datafile, undirected=True)
        walks = []
        with open(filesentences,"r") as fin:
            for line in fin.readlines():
                walk = line.strip().split(" ")
                walks.append(walk)
        return G, walks






