import torch
import torch.nn as nn
import torch.nn.functional as F

# 门控单元
# 作用：1. 序列深度建模 ； 2. 减轻梯度弥散，加速收敛
class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


class StockBlockLayer(nn.Module):
    def __init__(self, time_step, unit, multi_layer, stack_cnt=0):
        super(StockBlockLayer, self).__init__()
        self.time_step = time_step
        self.unit = unit
        self.stack_cnt = stack_cnt
        self.multi = multi_layer
        self.weight = nn.Parameter(
            torch.Tensor(1, 3 + 1, 1, self.time_step * self.multi,
                         self.multi * self.time_step))  # [K+1, 1, in_c, out_c]
        
        nn.init.xavier_normal_(self.weight) # 初始化
        self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi) # 12x5 -> 12x5
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step) # 12x5 -> 12

        if self.stack_cnt == 0:
            self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)
        # 数据原始特征表达    
        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step) # 线性变换，原始数据特征

        self.relu = nn.ReLU()
        self.GLUs = nn.ModuleList()
        self.output_channel = 4 * self.multi

        for i in range(3):
            if i == 0:
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
            elif i == 1:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
            else:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))

    def spe_seq_cell(self, input):
        # [32, 4, 1, 140, 12]
        batch_size, k, input_channel, node_cnt, time_step = input.size()
        input = input.view(batch_size, -1, node_cnt, time_step)
        
        # ffted = torch.rfft(input, 1, onesided=False) # [32, 4, 1, 140, 12, 2]
        ffted = torch.fft.fft(input, dim=-1) # torch.Size([32, 4, 140, 12])
        ffted_new = torch.stack((ffted.real, ffted.imag), -1)
        # print(ffted.size())
        # print(ffted_new.size())
        # print(ffted)
        # [32,140,4,12] -> [32,140, 48]
        real = ffted_new[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        img = ffted_new[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        # print(real.size())
        # print(img.size())
        # print('ok')

        for i in range(3):
            real = self.GLUs[i * 2](real)  # [32, 140, 240]
            img = self.GLUs[2 * i + 1](img)
        real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        # print(real.size())
        # print(img.size())
        # print('ok2')

        '''
        time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)
        print(time_step_as_inner.size()) # torch.Size([32, 4, 140, 60, 2])
        '''

        test = torch.complex(real, img)
        # print(test.size())

        # IDFT
        # iffted = torch.irfft(time_step_as_inner, 1, onesided=False)
        # iffted = torch.fft.ifft(time_step_as_inner, dim=-1) # torch.Size([32, 4, 140, 60, 2])
        iffted = torch.fft.ifft(test, dim=-1) # torch.Size([32, 4, 140, 60])
        # print(iffted.size())
        # print(iffted)

        # ffted_new = torch.stack((output.real, output.imag), -1)
        # iffted = abs(iffted)
        iffted = iffted.real # torch.Size([32, 4, 140, 60])

        return iffted

    def forward(self, x, mul_L):
        mul_L = mul_L.unsqueeze(1) # [4,1,140,140]
        x = x.unsqueeze(1) # [32,1,1,140,12]

        gfted = torch.matmul(mul_L, x)  # [32,4,1,140,12]

        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2) # torch.Size([32, 4, 1, 140, 60])

        igfted = torch.matmul(gconv_input, self.weight) # torch.Size([32, 4, 1, 140, 60])
        igfted = torch.sum(igfted, dim=1) # torch.Size([32, 1, 140, 60])

        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1)) # torch.Size([32, 140, 60])
        forecast = self.forecast_result(forecast_source) # # torch.Size([32, 140, 12])

        if self.stack_cnt == 0:
            backcast_short = self.backcast_short_cut(x).squeeze(1) # torch.Size([32, 1, 140, 12])
            backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short) # torch.Size([32, 1, 140, 12])
        else:
            backcast_source = None

        return forecast, backcast_source


class Model(nn.Module):
    def __init__(self, units, stack_cnt, time_step, multi_layer, horizon=1, dropout_rate=0.5, leaky_rate=0.2,
                 device='cpu'):
        super(Model, self).__init__()
        self.unit = units # 输出的特征维度 140
        # StemGNN block
        self.stack_cnt = stack_cnt  # block的个数
        self.alpha = leaky_rate

        self.time_step = time_step # windows size
        self.horizon = horizon # 预测长度

        # 注意力权重参数
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1))) # K
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414) # 初始化
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1))) # Q
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414) # 初始化

        self.GRU = nn.GRU(self.time_step, self.unit) # GRU模块 12，140
        self.multi_layer = multi_layer # 多头层数

        self.stock_block = nn.ModuleList()
        self.stock_block.extend(
            [StockBlockLayer(self.time_step, self.unit, self.multi_layer, stack_cnt=i) for i in range(self.stack_cnt)])
        self.fc = nn.Sequential(
            nn.Linear(int(self.time_step), int(self.time_step)),
            nn.LeakyReLU(),
            nn.Linear(int(self.time_step), self.horizon),
        )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.to(device)

    def get_laplacian(self, graph, normalize):
        """
        return the laplacian of the graph.
        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.
        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N] 140
        laplacian = laplacian.unsqueeze(0) # [1x140x140]
        first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float) # 零矩阵
        second_laplacian = laplacian # [1x140x140]
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian # # [1x140x140]
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian # # [1x140x140]
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0) # [4, 140, 140]
        return multi_order_laplacian # [4x140x140]


    def latent_correlation_layer(self, x):
        input, _ = self.GRU(x.permute(2, 0, 1).contiguous()) # # 维度变换 [140, 32, 12] => [140, 32, 140]
        input = input.permute(1, 0, 2).contiguous() # 维度变换 [32, 140, 140]
        attention = self.self_graph_attention(input)
        attention = torch.mean(attention, dim=0)

        degree = torch.sum(attention, dim=1) # 度向量
        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T) # 对称阵
        degree_l = torch.diag(degree) # 度矩阵
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7)) # 开根
        # D-1/2 * A * D-1/2 
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat)) # [140, 140] # 拉普拉斯矩阵
        mul_L = self.cheb_polynomial(laplacian) # [4, 140, 140] # 拉普拉斯矩阵 => 切比雪夫多项式，多阶

        return mul_L, attention # 多阶拉普拉斯矩阵，注意力矩阵


    def self_graph_attention(self, input): # input = [32, 140, 140] batch, seq, size
        input = input.permute(0, 2, 1).contiguous() # [32, 140, 140] batch, size, seq
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key) # [32, 140, 1]
        query = torch.matmul(input, self.weight_query) # # [32, 140, 1]
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1) # # [32, 140*140, 1]
        data = data.squeeze(2) # [32, 140*140]
        data = data.view(bat, N, -1) # [32, 140, 140]
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    def graph_fft(self, input, eigenvectors):
        return torch.matmul(eigenvectors, input)

    def forward(self, x):
        # part1
        mul_L, attention = self.latent_correlation_layer(x) # 多阶拉普拉斯矩阵，注意力 # [4x140x140]， [140x140]
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous() # [32, 1, 140, 12]

        # part2
        result = []
        for stack_i in range(self.stack_cnt):
            # X Shape: torch.size[32, 1, 140, 12]
            # mul_L Shape: torch.size[4x140x140]
            forecast, X = self.stock_block[stack_i](X, mul_L)
            result.append(forecast)
        
        forecast = result[0] + result[1]
        forecast = self.fc(forecast)
        
        if forecast.size()[-1] == 1:
            return forecast.unsqueeze(1).squeeze(-1), attention
        else:
            return forecast.permute(0, 2, 1).contiguous(), attention
