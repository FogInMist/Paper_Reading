import torch
import torch.nn as nn
from layer import *

class HeteGAT_multi(nn.Module):
    def __init__(self, inputs_list, nb_classes, nb_nodes, attn_drop, ffd_drop,
                 bias_mat_list, hid_units, n_heads, activation=nn.ELU(), residual=False):
        super(HeteGAT_multi, self).__init__()
        self.inputs_list = inputs_list  # 输入列表
        self.nb_classes = nb_classes  # 类别数
        self.nb_nodes = nb_nodes  # 节点数
        self.attn_drop = attn_drop # 注意力权重衰减率
        self.ffd_drop = ffd_drop  # 前馈神经网络权重衰减率
        self.bias_mat_list = bias_mat_list  # 偏置矩阵列表
        self.hid_units = hid_units  # 隐藏单元数
        self.n_heads = n_heads  # 注意力头数
        self.activation = activation  # 激活函数
        self.residual =residual  # 残差标志
        self.mp_att_size = 128
        self.layers = self._make_attn_head()    # 用于创建注意力头
        self.simpleAttLayer = SimpleAttLayer(64, self.mp_att_size, time_major=False,return_alphas=True)
        self.fc = nn.Linear(64, self.nb_classes)
        
    def _make_attn_head(self):  # 用于创建注意力头。包括一个循环，遍历输入列表和偏置矩阵列表，为每个输入创建一个注意力头。
        layers = []
        for inputs,biases in zip(self.inputs_list, self.bias_mat_list):
            layers.append(Attn_head(in_channel=inputs.shape[1], out_sz=self.hid_units[0], bias_mat=biases, in_drop=self.ffd_drop,coef_drop=self.attn_drop,activation=self.activation,residual=self.residual))
        #print("当前有{}个注意力头".format(len(layers)))
        
        return nn.Sequential(*list(m for m in layers))
        
    def forward(self,x):
        embed_list = []
        for i,(inputs,biases) in enumerate(zip(x, self.bias_mat_list)):
            attns = []
            jhy_embeds = []
            for _ in range(self.n_heads[0]):
                attns.append(self.layers[i](inputs))
            h_1 = torch.cat(attns,dim=1)
            #print("h_1.shape:",h_1.shape)
            #print("torch.squeeze(h_1).shape",torch.squeeze(h_1).shape)
            #print("torch.squeeze(h_1).reshape(h_1.shap[-1],1,-1.shape)",torch.squeeze(h_1).reshape(h_1.shape[-1],1,-1).shape)
            embed_list.append(torch.squeeze(h_1).reshape(h_1.shape[-1],1,-1))

        multi_embed = torch.cat(embed_list,dim=1)
        #print("multi_embed.shape:",multi_embed.shape)
        final_embed,att_val = self.simpleAttLayer(multi_embed)
        out = [] 

        for i in range(self.n_heads[-1]):
           out.append(self.fc(final_embed))
        #print("out[0].shape:",out[0].shape)
        
        return out[0]
