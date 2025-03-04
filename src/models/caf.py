import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.utils import negative_sampling

class CAF(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, gc_layer, bn):
        super().__init__()

        self.conv1 = SAGEConv(in_dim, hidden_dim, normalize=True)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout)
        )
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, normalize=True)
        self.conv2.aggr = 'mean'
        self.fc = nn.Linear(int(hidden_dim/2), out_dim)
        self.sens_fc = nn.Linear(int(hidden_dim/2), 1)

        self.indices1 = None
        self.indices2 = None

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
        embed = self.conv2(x, edge_index)
        preds = self.fc(embed[:,:int(embed.shape[1]/2)])
        return preds.squeeze(), embed

    def sens_pred(self, embed):
        sens = self.sens_fc(embed[:,int(embed.shape[1]/2):])
        return sens
    
    def loss(self, out, batch, mode, retrain, epoch, retrain_config):
        tepoch, causal_coeff, disentangle_coeff, rec_coeff, dist_mode, indices_num = retrain_config['tepoch'], retrain_config['causal_coeff'], retrain_config['disentangle_coeff'], retrain_config['rec_coeff'], retrain_config['dist_mode'], retrain_config['indices_num']
        preds, embed = out 
        labels, sens, mask = batch['y'], batch['sens'], batch[f'{mode}_mask']
        loss_pred = self.calculate_prediction_loss(preds, labels, mask)
        if retrain == False:
            return loss_pred
        else:
            if (epoch % tepoch) == 0:
                indices1, indices2 = self.find_counterfactuals(embed, preds, sens, indices_num)
                self.indices1, self.indices2 = indices1, indices2
            causal_loss, style_loss = self.calculate_causal_loss(embed, self.indices1, self.indices2, dist_mode, indices_num)
            disentangle_loss = self.calculate_disentangle_loss(embed)
            rec_loss = self.calculate_reconstruction_loss(embed, batch['edge_index'])
            # sens_loss = self.calculate_sens_loss(embed, sens)
            sens_loss = 0
            loss_total = loss_pred + causal_coeff * (causal_loss + style_loss) + disentangle_coeff * disentangle_loss + rec_coeff * rec_loss + sens_loss
            return loss_total

    def calculate_prediction_loss(self, preds, labels, mask):
        if preds.dim() == 1:
            return F.binary_cross_entropy_with_logits(preds[mask], labels[mask].float())
        elif preds.dim() == 2:
            return F.cross_entropy(preds[mask], labels[mask].float())
        
    def calculate_causal_loss(self, embed, indices1, indices2, dist_mode, indices_num):
        half_dim = int(embed.shape[1]/2)
        causal_embed, style_embed = embed[:, :half_dim], embed[:, half_dim:]
        causal_loss, style_loss = 0, 0
        for i in range(indices_num):
            if dist_mode == 'L1':
                causal_loss +=  torch.mean(torch.abs(causal_embed - causal_embed[indices1[:,i], :]))
                style_loss +=  torch.mean(torch.abs(style_embed - style_embed[indices2[:,i], :]))
            elif dist_mode == 'L2':
                causal_loss +=  torch.mean((causal_embed - causal_embed[indices1[:,i], :])**2)
                style_loss +=  torch.mean((style_embed - style_embed[indices2[:,i], :])**2)
            elif dist_mode == 'L1+L2':
                causal_loss +=  torch.mean(torch.abs(causal_embed - causal_embed[indices1[:,i], :])) + torch.mean((causal_embed - causal_embed[indices1[:,i], :])**2)
                style_loss +=  torch.mean(torch.abs(style_embed - style_embed[indices2[:,i], :])) + torch.mean((style_embed - style_embed[indices2[:,i], :])**2)
            elif dist_mode == 'cosine':
                causal_loss +=  1-F.cosine_similarity(causal_embed, causal_embed[indices1[:,i], :]).mean()
                style_loss +=  1-F.cosine_similarity(style_embed, style_embed[indices2[:,i], :]).mean()
            elif dist_mode == 'abscosine':
                causal_loss +=  1-torch.abs(F.cosine_similarity(causal_embed, causal_embed[indices1[:,i], :])).mean()
                style_loss +=  1-torch.abs(F.cosine_similarity(style_embed, style_embed[indices2[:,i], :])).mean()
        return causal_loss, style_loss
    
    def calculate_disentangle_loss(self, embed):
        half_dim = int(embed.shape[1]/2)
        causal_embed, style_embed = embed[:, :half_dim], embed[:, half_dim:]
        disentangle_loss = torch.abs(F.cosine_similarity(causal_embed, style_embed)).mean()
        return disentangle_loss
    
    def calculate_reconstruction_loss(self, embed, edge_index):
        decoder = InnerProductDecoder()
        pos_loss = -torch.log(decoder(embed, edge_index) + 1e-15).mean()
        neg_edge_index = negative_sampling(edge_index, num_nodes=embed.shape[0], num_neg_samples=edge_index.shape[1])
        neg_loss = -torch.log(1 - decoder(embed, neg_edge_index) + 1e-15).mean()
        rec_loss = pos_loss + neg_loss
        return rec_loss
    
    def calculate_sens_loss(self, embed, sens):
        sens_output = self.sens_pred(embed)
        sens_loss = F.binary_cross_entropy_with_logits(sens_output, sens.unsqueeze(1).float())
        return sens_loss

    def find_counterfactuals(self, embed, preds, sens, indices_num):
        embed = embed.detach()
    
        # Compute label_pair and sens_pair
        pseudo_label = preds.reshape(-1, 1).sigmoid().bool()
        sens_torch = sens.reshape(-1, 1).sigmoid().bool()
        label_pair = pseudo_label.eq(pseudo_label.t())
        sens_pair = sens_torch.eq(sens_torch.t())

        # Compute select_adj and select_adj2
        select_adj = ~label_pair & sens_pair
        select_adj2 = label_pair & ~sens_pair

        # Compute the normalized distance
        distance = torch.cdist(embed, embed, p=2)
        distance = (distance - distance.min()) / (distance.max() - distance.min())

        # Find indices
        indices1 = self.find_indices(distance.cpu(), select_adj2, indices_num)
        indices2 = self.find_indices(distance.cpu(), select_adj, indices_num)
        
        return indices1, indices2


    def find_indices(self, distance, select_adj, num_indices):
        distance = distance.clone()
        distance[~select_adj] = float('inf')
        _, indices = torch.topk(distance, num_indices, largest=False)
        return indices
