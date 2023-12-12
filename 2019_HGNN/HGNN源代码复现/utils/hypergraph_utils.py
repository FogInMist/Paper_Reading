# --------------------------------------------------------
# Utility functions for Hypergraph
#
# Author: Yifan Feng
# Date: November 2018
# --------------------------------------------------------
import numpy as np


def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)  # 转换成np格式
    aa = np.sum(np.multiply(x, x), 1)  # 矩阵对应位置相乘，按行求和; sum(x^2)
    ab = x * x.T  # mat下，表示内积。array下，表示对应位置相乘; X11*X12 + X12*X22 + ...
    dist_mat = aa + aa.T - 2 * ab  # aa + aa.T = (12311, 12311); 欧式距离^2
    dist_mat[dist_mat < 0] = 0  # 计算时，会出现极小的负数
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def feature_concat(*F_list, normal_col=False):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
    features = None
    for f in F_list:  # (fts, gvcnn_ft)
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])  # 转换成维度为2
            # normal each column
            if normal_col:  # 列正则化
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:  # 输入的组合(null, 邻居距离矩阵)
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))  # 邻居矩阵拼接
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)  # 超边矩阵
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)  # 超边权重矩阵
    # the degree of the node
    DV = np.sum(H * W, axis=1)  # 节点度; (12311,)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)  # 超边的度; (24622,)

    invDE = np.mat(np.diag(np.power(DE, -1)))  # DE^-1; 建立对角阵
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))  # DV^-1/2
    W = np.mat(np.diag(W))  # 超边权重矩阵
    H = np.mat(H)  # 超边矩阵
    HT = H.T

    if variable_weight:  # 超边权重是否可变
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix; 由超图节点距离矩阵构造超图关联矩阵
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))  # (12311, 12311)的0矩阵
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0  # 自己和自己距离为0;
        dis_vec = dis_mat[center_idx]  # 这个节点的邻居节点距离
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()  # 元素从小到大排列，提取其对应的索引; 距离最近的节点
        avg_dis = np.average(dis_vec)  # 取均值
        if not np.any(nearest_idx[:k_neig] == center_idx):  # 如果center_idx不在前k个中，将最后一位赋值为center_idx; 绝大部分存在，除非其他距离为0的很多
            nearest_idx[k_neig - 1] = center_idx
        # 构建超边;
        for node_idx in nearest_idx[:k_neig]:  # 距离最近的10个节点
            if is_probH:  # dis_距离越小，计算结果越接近1;  按列进行构建超边
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)  # exp(-dis^2/dis_avg^2)
            else:
                H[node_idx, center_idx] = 1.0  # 距离结果直接赋值为1;
    return H


def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:  # 如果维度不等于2，进行维度转换
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:  # 如果是int，转换成list(int)
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)  # 计算节点两两之间的欧式距离; 即(X1 - X2)^2
    H = []
    for k_neig in K_neigs:  # 遍历如多个近邻超边
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)  # 距离最近的n个邻居的距离矩阵;
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)  # 超边concat操作; 如果计算多个近邻超边
        else:
            H.append(H_tmp)
    return H  # n个邻居中距离矩阵
