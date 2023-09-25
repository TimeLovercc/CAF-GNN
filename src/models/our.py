import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn.utils import spectral_norm


class Our(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Our, self).__init__()
        self.body = GCN_Body(nfeat,nhid,dropout)
        self.fc = nn.Linear(int(nhid/2), nclass)
        self.sens_fc = nn.Linear(int(nhid/2), 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        embed = self.body(x, edge_index)
        # batch normalization
        x = self.fc(embed[:,:int(embed.shape[1]/2)])
        return x, embed
        # embed1 = self.body(x, edge_index)
        # embed2 = self.proj(embed1)
        # embed3 = self.pred(embed2)
        # x = self.fc(embed1[:,:int(embed1.shape[1]/2)])
        # return x, embed1, embed2, embed3


    def sens_pred(self, embed):
        sens = self.sens_fc(embed[:,int(embed.shape[1]/2):])
        return sens

class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.bn = nn.BatchNorm1d(nhid)
        # self.gc2 = GCNConv(nhid, nhid)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        # x = F.relu(x) 
        x = self.bn(x)
        # x = self.gc2(x, edge_index)
        return x    




