import torch
import utils as u
import numpy as np
import time

ECOLS = u.Namespace({'source': 0,
                     'target': 1,
                     'time': 2,
                     'label':3}) #--> added for edge_cls

# def get_2_hot_deg_feats(adj,max_deg_out,max_deg_in,num_nodes):
#     #For now it'll just return a 2-hot vector
#     adj['vals'] = torch.ones(adj['idx'].size(0))
#     degs_out, degs_in = get_degree_vects(adj,num_nodes)
    
#     degs_out = {'idx': torch.cat([torch.arange(num_nodes).view(-1,1),
#                                   degs_out.view(-1,1)],dim=1),
#                 'vals': torch.ones(num_nodes)}
    
#     # print ('XXX degs_out',degs_out['idx'].size(),degs_out['vals'].size())
#     degs_out = u.make_sparse_tensor(degs_out,'long',[num_nodes,max_deg_out])

#     degs_in = {'idx': torch.cat([torch.arange(num_nodes).view(-1,1),
#                                   degs_in.view(-1,1)],dim=1),
#                 'vals': torch.ones(num_nodes)}
#     degs_in = u.make_sparse_tensor(degs_in,'long',[num_nodes,max_deg_in])

#     hot_2 = torch.cat([degs_out,degs_in],dim = 1)
#     hot_2 = {'idx': hot_2._indices().t(),
#              'vals': hot_2._values()}

#     return hot_2

def get_1_hot_deg_feats(adj,max_deg,num_nodes):
    # For now it'll just return a 2-hot vector
    new_vals = torch.ones(adj['idx'].size(0))  # 数据量全1矩阵
    new_adj = {'idx':adj['idx'], 'vals': new_vals}  # 新的vals
    degs_out, _ = get_degree_vects(new_adj,num_nodes)  # 获得邻接矩阵的out-degree
    
    degs_out = {'idx': torch.cat([torch.arange(num_nodes).view(-1,1),  # 构建节点和对应的出度; 构建节点degree的one-hot特征;
                                  degs_out.view(-1,1)],dim=1),
                'vals': torch.ones(num_nodes)}
    
    # print ('XXX degs_out',degs_out['idx'].size(),degs_out['vals'].size())
    degs_out = u.make_sparse_tensor(degs_out,'long',[num_nodes,max_deg])  # 构建节点和出度的稀疏矩阵; 1000*162

    hot_1 = {'idx': degs_out._indices().t(),
             'vals': degs_out._values()}
    return hot_1  # 节点和degree的矩阵; val===1

def get_max_degs(args,dataset,all_window=False):
    max_deg_out = []
    max_deg_in = []
    for t in range(dataset.min_time, dataset.max_time):
        if all_window:
            window = t+1
        else:
            window = args.adj_mat_time_window  # 时间窗口
        # 获得满足时间要求的点边数据
        cur_adj = get_sp_adj(edges = dataset.edges,  # 点边数据
                             time = t,  # 时间t
                             weighted = False,
                             time_window = window)  # 时间窗口
        # print(window)
        cur_out, cur_in = get_degree_vects(cur_adj,dataset.num_nodes)  # 出度和入度
        max_deg_out.append(cur_out.max())  # outdegree最大值
        max_deg_in.append(cur_in.max())  # indegree最大值
        # max_deg_out = torch.stack([max_deg_out,cur_out.max()]).max()
        # max_deg_in = torch.stack([max_deg_in,cur_in.max()]).max()
    # exit()
    max_deg_out = torch.stack(max_deg_out).max()  # 全体最大值
    max_deg_in = torch.stack(max_deg_in).max()  # 全体最小值
    max_deg_out = int(max_deg_out) + 1  # 自连接
    max_deg_in = int(max_deg_in) + 1
    
    return max_deg_out, max_deg_in

def get_max_degs_static(num_nodes, adj_matrix):
    cur_out, cur_in = get_degree_vects(adj_matrix, num_nodes)
    max_deg_out = int(cur_out.max().item()) + 1
    max_deg_in = int(cur_in.max().item()) + 1
    
    return max_deg_out, max_deg_in


def get_degree_vects(adj,num_nodes):
    adj = u.make_sparse_tensor(adj,'long',[num_nodes])  # 构建符合类型的点边稀疏矩阵
    degs_out = adj.matmul(torch.ones(num_nodes,1,dtype = torch.long))  # out-degree; 出度
    degs_in = adj.t().matmul(torch.ones(num_nodes,1,dtype = torch.long))  # in-degree; 入度
    return degs_out, degs_in

def get_sp_adj(edges,time,weighted,time_window):
    idx = edges['idx']  # 索引
    subset = idx[:,ECOLS.time] <= time  # 选择满足时间的数据
    subset = subset * (idx[:,ECOLS.time] > (time - time_window))  # 还要满足时间窗口
    idx = edges['idx'][subset][:,[ECOLS.source, ECOLS.target]]   # 满足条件的点边数据
    vals = edges['vals'][subset]  # label===1
    out = torch.sparse.FloatTensor(idx.t(),vals).coalesce()  # 构建满足条件的稀疏矩阵
    
    
    idx = out._indices().t()  # src-dst 边
    if weighted:
        vals = out._values()  # 边的权重
    else:  # weighted=False，重新赋值weight
        vals = torch.ones(idx.size(0),dtype=torch.long)  # vals的值重新赋值为1

    return {'idx': idx, 'vals': vals}

def get_edge_labels(edges,time):
    idx = edges['idx']
    subset = idx[:,ECOLS.time] == time
    idx = edges['idx'][subset][:,[ECOLS.source, ECOLS.target]]  
    vals = edges['idx'][subset][:,ECOLS.label]

    return {'idx': idx, 'vals': vals}


def get_node_mask(cur_adj,num_nodes):
    mask = torch.zeros(num_nodes) - float("Inf")
    non_zero = cur_adj['idx'].unique()  # 图中所有节点

    mask[non_zero] = 0
    
    return mask

def get_static_sp_adj(edges,weighted):
    idx = edges['idx']
    #subset = idx[:,ECOLS.time] <= time
    #subset = subset * (idx[:,ECOLS.time] > (time - time_window))

    #idx = edges['idx'][subset][:,[ECOLS.source, ECOLS.target]]  
    if weighted:
        vals = edges['vals'][subset]
    else:
        vals = torch.ones(idx.size(0),dtype = torch.long)

    return {'idx': idx, 'vals': vals}

def get_sp_adj_only_new(edges,time,weighted):
    return get_sp_adj(edges, time, weighted, time_window=1)

def normalize_adj(adj,num_nodes):
    '''
    takes an adj matrix as a dict with idx and vals and normalize it by: 
        - adding an identity matrix, 
        - computing the degree vector
        - multiplying each element of the adj matrix (aij) by (di*dj)^-1/2
    '''
    idx = adj['idx']
    vals = adj['vals']

    # adj的稀疏矩阵
    sp_tensor = torch.sparse.FloatTensor(idx.t(),vals.type(torch.float),torch.Size([num_nodes,num_nodes]))
    # eye矩阵
    sparse_eye = make_sparse_eye(num_nodes)  # 单位阵
    sp_tensor = sparse_eye + sp_tensor  # 自连接边

    idx = sp_tensor._indices()
    vals = sp_tensor._values()

    degree = torch.sparse.sum(sp_tensor,dim=1).to_dense()  # degree
    di = degree[idx[0]]  # src节点对应的度
    dj = degree[idx[1]]  # dst节点对应的度

    vals = vals * ((di * dj) ** -0.5)
    
    return {'idx': idx.t(), 'vals': vals}

def make_sparse_eye(size):
    eye_idx = torch.arange(size)
    eye_idx = torch.stack([eye_idx,eye_idx],dim=1).t()  # 自连接矩阵
    vals = torch.ones(size)
    eye = torch.sparse.FloatTensor(eye_idx,vals,torch.Size([size,size]))
    return eye

def get_all_non_existing_edges(adj,tot_nodes):
    true_ids = adj['idx'].t().numpy()
    true_ids = get_edges_ids(true_ids,tot_nodes)

    all_edges_idx = np.arange(tot_nodes)
    all_edges_idx = np.array(np.meshgrid(all_edges_idx,
                                         all_edges_idx)).reshape(2,-1)

    all_edges_ids = get_edges_ids(all_edges_idx,tot_nodes)

    #only edges that are not in the true_ids should keep here
    mask = np.logical_not(np.isin(all_edges_ids,true_ids))

    non_existing_edges_idx = all_edges_idx[:,mask]
    edges = torch.tensor(non_existing_edges_idx).t()
    vals = torch.zeros(edges.size(0), dtype = torch.long)
    return {'idx': edges, 'vals': vals}


def get_non_existing_edges(adj,number, tot_nodes, smart_sampling, existing_nodes=None):
    # print('----------')
    t0 = time.time()
    idx = adj['idx'].t().numpy()
    true_ids = get_edges_ids(idx,tot_nodes)  # src*1000 + dst; src节点*节点数量 + dst节点

    true_ids = set(true_ids)  # 有没有重复的节点

    # the maximum of edges would be all edges that don't exist between nodes that have edges; 所有有边的节点之间不存在连接边
    num_edges = min(number,idx.shape[1] * (idx.shape[1]-1) - len(true_ids))  # (负采样数量, 可能存在的);

    if smart_sampling:
        #existing_nodes = existing_nodes.numpy()
        def sample_edges(num_edges):
            # print('smart_sampling')
            from_id = np.random.choice(idx[0],size = num_edges,replace = True)  # 边采样, src节点采样
            to_id = np.random.choice(existing_nodes,size = num_edges, replace = True)  # 节点采样, dst节点采样
            #print ('smart_sampling', from_id, to_id)
            
            if num_edges>1:
                edges = np.stack([from_id,to_id])  # (2, 20496400)
            else:
                edges = np.concatenate([from_id,to_id])
            return edges
    else:
        def sample_edges(num_edges):
            if num_edges > 1:
                edges = np.random.randint(0,tot_nodes,(2,num_edges))
            else:
                edges = np.random.randint(0,tot_nodes,(2,))
            return edges

    edges = sample_edges(num_edges*4)  # num_edges:负采样的边; 采样之后得到的节点和边

    edge_ids = edges[0] * tot_nodes + edges[1]  # 负采样的样本，去重操作
    
    out_ids = set()
    num_sampled = 0
    sampled_indices = []
    for i in range(num_edges*4):  # 选择出不是自连接或者已经存在的边; 负采样
        eid = edge_ids[i]  # 负采样样本;
        #ignore if any of these conditions happen; 1.负采样已经出现 2. 自连接 3. 正样本中已经存在的边
        if eid in out_ids or edges[0,i] == edges[1,i] or eid in true_ids:
            continue

        #add the eid and the index to a list
        out_ids.add(eid)
        sampled_indices.append(i)  # 采样索引
        num_sampled += 1

        #if we have sampled enough edges break; 采样数量和正样本相同
        if num_sampled >= num_edges:
            break

    edges = edges[:,sampled_indices]
    edges = torch.tensor(edges).t()
    vals = torch.zeros(edges.size(0),dtype = torch.long)  # label==0
    return {'idx': edges, 'vals': vals}

def get_edges_ids(sp_idx, tot_nodes):
    # print(sp_idx)
    # print(tot_nodes)
    # print(sp_idx[0]*tot_nodes)
    return sp_idx[0]*tot_nodes + sp_idx[1]  # src*1000 + dst


