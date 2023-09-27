import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class CAF(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout): 
        super().__init__()
        self.conv1 = SAGEConv(nfeat, nhid, normalize=True)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            nn.Dropout(p=dropout)
        )
        self.conv2 = SAGEConv(nhid, nhid, normalize=True)
        self.conv2.aggr = 'mean'
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
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        embed = self.conv2(x, edge_index)
        return self.fc(embed[:,:int(embed.shape[1]/2)]), embed

    def sens_pred(self, embed):
        sens = self.sens_fc(embed[:,int(embed.shape[1]/2):])
        return sens
