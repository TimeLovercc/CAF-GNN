import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, gc_layer, bn):
        super().__init__()
        self.gc_layer = gc_layer
        self.p = dropout
        self.convs = nn.ModuleList([GCNConv(in_dim if i == 0 else hidden_dim, hidden_dim) for i in range(gc_layer)])
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

    
