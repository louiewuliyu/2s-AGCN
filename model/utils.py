import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch.nn import Parameter

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class SE_Temporal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, temporal_size=300):
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.se_temporal = nn.Sequential(nn.Linear(temporal_size, temporal_size, bias=False), nn.Sigmoid())

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(1, 0),
                              stride=(stride, 1), dilation=(1, 1))
        self.conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(2, 0),
                              stride=(stride, 1), dilation=(2, 1))
        self.conv_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(3, 0),
                              stride=(stride, 1), dilation=(3, 1))
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.bn_3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.temporal_size = temporal_size
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.view(N*C, T, V)

        for i in range(self.temporal_size-1):
            y_i = (x[:, i+1, :] - x[:, i, :]).unsqueeze(1)
            y = y_i if i == 0 else torch.cat((y, y_i), dim=1)

        y = self.avg_pool(y).view(N*C, T)    # (NC, T, 1)
        y = self.se_temporal(y).view(N*C, T, 1)
        x = x * y.expand_as(x)  # (NC, T, V)

        x = x.view(N, C, T, V)
        x1 = self.conv_1(x)
        x1 = self.bn_1(x1)
        x2 = self.conv_2(x)
        x2 = self.bn_2(x2)
        x3 = self.conv_3(x)
        x3 = self.bn_3(x3)

        x = x1 + x2 + x3
        return x

class SAGPool(torch.nn.Module):
    def __init__(self,in_channels, A, ratio=0.5, Conv=GCNConv, non_linearity=torch.tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
        self.A = A

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        if batch is None:
            batch = self.A.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x, self.A).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm

class Level_Structure(nn.Module):
    def __init__(self, in_channels, A, ratio=0.5):
        self.sag_pool = SAGPool(in_channels, A, ratio)
