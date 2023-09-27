import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, gc_layer, bn):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, normalize=True)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout)
        )
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, normalize=True)
        self.conv2.aggr = 'mean'
        self.fc = nn.Linear(hidden_dim, out_dim)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, batch):
        x, edge_index = batch['x'], batch['edge_index'] 
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return x.squeeze()

    def loss(self, out, batch, mode):
        preds = out
        labels, mask = batch['y'], batch[f'{mode}_mask']
        if preds.dim() == 1:
            return F.binary_cross_entropy_with_logits(preds[mask], labels[mask].float())
        elif preds.dim() == 2:
            return F.cross_entropy(preds[mask], labels[mask])
