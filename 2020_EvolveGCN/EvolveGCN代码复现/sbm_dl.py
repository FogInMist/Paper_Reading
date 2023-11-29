import torch
import utils as u
import os

class sbm_dataset():
    def __init__(self,args):
        assert args.task in ['link_pred'], 'sbm only implements link_pred'
        self.ecols = u.Namespace({'FromNodeId': 0,  # 定义dictionary
                                  'ToNodeId': 1,
                                  'Weight': 2,
                                  'TimeStep': 3
                                })
        args.sbm_args = u.Namespace(args.sbm_args)  # 文件相关dictionary

        #build edge data structure
        edges = self.load_edges(args.sbm_args)  # 导入数据
        timesteps = u.aggregate_by_time(edges[:,self.ecols.TimeStep], args.sbm_args.aggr_time)  # 计算时间间隔;
        self.max_time = timesteps.max()
        self.min_time = timesteps.min()
        print ('TIME', self.max_time, self.min_time )
        edges[:,self.ecols.TimeStep] = timesteps  # 重新复制time

        edges[:,self.ecols.Weight] = self.cluster_negs_and_positives(edges[:,self.ecols.Weight])  # weight权重赋值0, 1
        self.num_classes = edges[:,self.ecols.Weight].unique().size(0)  # 类别数

        self.edges = self.edges_to_sp_dict(edges)  # 数据处理: {idx, vals}
        
        #random node features
        self.num_nodes = int(self.get_num_nodes(edges))  # 节点数量
        self.feats_per_node = args.sbm_args.feats_per_node  # 节点特征数量，预先定义; 后面没有使用
        self.nodes_feats = torch.rand((self.num_nodes,self.feats_per_node))  # 随机赋予节点特征

        self.num_non_existing = self.num_nodes ** 2 - edges.size(0)  # 节点两两可能形成的边数量 - 现在存在边的数量

    def cluster_negs_and_positives(self,ratings):
        pos_indices = ratings >= 0
        neg_indices = ratings < 0
        ratings[pos_indices] = 1
        ratings[neg_indices] = 0
        return ratings

    def prepare_node_feats(self,node_feats):
        node_feats = node_feats[0]
        return node_feats

    def edges_to_sp_dict(self,edges):
        idx = edges[:,[self.ecols.FromNodeId,  # src: 0
                       self.ecols.ToNodeId,    # dst: 1
                       self.ecols.TimeStep]]   # time: 3

        vals = edges[:,self.ecols.Weight]  # weight: 2
        return {'idx': idx,
                'vals': vals}

    def get_num_nodes(self,edges):
        all_ids = edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]  # src, dst
        num_nodes = all_ids.max() + 1  # 最大节点index（节点从0开始）
        return num_nodes

    def load_edges(self,sbm_args, starting_line = 1):
        file = os.path.join(sbm_args.folder,sbm_args.edges_file)
        with open(file) as f:
            lines = f.read().splitlines()  # 导入数据
        edges = [[float(r) for r in row.split(',')] for row in lines[starting_line:]]  # 从第1个开始是数据; 遍历数据，按,分割
        edges = torch.tensor(edges,dtype = torch.long)
        return edges

    def make_contigous_node_ids(self,edges):
        new_edges = edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]
        _, new_edges = new_edges.unique(return_inverse=True)
        edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]] = new_edges
        return edges
