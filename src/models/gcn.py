import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, gc_layer, bn):
        super().__init__()
        self.gc_layer = gc_layer
        self.p = dropout
        self.convs = nn.ModuleList()
        if gc_layer == 1:
            self.convs.append(GCNConv(in_dim, out_dim))
        elif gc_layer == 2:
            self.convs.append(GCNConv(in_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, out_dim))
        else:
            self.convs.append(GCNConv(in_dim, hidden_dim))
            for layer in range(layer-2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, out_dim))
        if bn == True:
            self.bns = nn.ModuleList()
            for layer in range(layer-1):
                self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, batch):
        x, edge_index = batch['x'], batch['edge_index']
        x = F.dropout(x, p=self.p, training=self.training)
        for layer in range(self.gc_layer-1):
            x = self.convs[layer](x, edge_index)
            if hasattr(self, 'bns'):
                x = self.bns[layer](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.p, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x.squeeze()
    
    def loss(self, out, batch, mode):
        preds = out
        labels, mask = batch['y'], batch[f'{mode}_mask']
        if preds.dim() == 1:
            return F.binary_cross_entropy_with_logits(preds[mask], labels[mask].float())
        elif preds.dim() == 2:
            return F.cross_entropy(preds[mask], labels[mask])

    
