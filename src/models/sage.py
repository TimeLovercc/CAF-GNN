import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class SAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, gc_layer, bn):
        super().__init__()
        self.gc_layer = gc_layer
        self.p = dropout
        self.convs = nn.ModuleList([SAGEConv(in_dim if i == 0 else hidden_dim, hidden_dim) for i in range(gc_layer)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(gc_layer - 1)]) if bn else None
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, batch):
        x, edge_index = batch['x'], batch['edge_index']
        x = F.dropout(x, p=self.p, training=self.training)
        for layer in range(self.gc_layer-1):
            x = self.convs[layer](x, edge_index)
            if self.bns is not None:
                x = self.bns[layer](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.convs[-1](x, edge_index))
        if self.bns is not None:
            x = self.bns[-1](x)
        x = self.fc(x)
        return x.squeeze()
    
    def loss(self, out, batch, mode):
        preds = out
        labels, mask = batch['y'], batch[f'{mode}_mask']
        if preds.dim() == 1:
            return F.binary_cross_entropy_with_logits(preds[mask], labels[mask].float())
        elif preds.dim() == 2:
            return F.cross_entropy(preds[mask], labels[mask])




# class SAGE(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout): 
#         super(SAGE, self).__init__()
#         self.conv1 = SAGEConv(nfeat, nhid, normalize=True)
#         self.conv1.aggr = 'mean'
#         self.transition = nn.Sequential(
#             nn.ReLU(),
#             nn.BatchNorm1d(nhid),
#             nn.Dropout(p=dropout)
#         )
#         self.conv2 = SAGEConv(nhid, nhid, normalize=True)
#         self.conv2.aggr = 'mean'
#         self.fc = nn.Linear(nhid, nclass)

#         for m in self.modules():
#             self.weights_init(m)

#     def weights_init(self, m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)

#     def forward(self, x, edge_index): 
#         x = self.conv1(x, edge_index)
#         x = self.transition(x)
#         x = self.conv2(x, edge_index)
#         return self.fc(x)

# class SAGE_Body(nn.Module):
#     def __init__(self, nfeat, nhid, dropout):
#         super(SAGE_Body, self).__init__()
#         self.conv1 = SAGEConv(nfeat, nhid, normalize=True)
#         self.conv1.aggr = 'mean'
#         self.transition = nn.Sequential(
#             nn.ReLU(),
#             nn.BatchNorm1d(nhid),
#             nn.Dropout(p=dropout)
#         )
#         self.conv2 = SAGEConv(nhid, nhid, normalize=True)
#         self.conv2.aggr = 'mean'

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = self.transition(x)
#         x = self.conv2(x, edge_index)
#         return x