import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class CAF(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, gc_layer, bn, tepoch):
        super().__init__()
        self.tepoch = tepoch

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
        return self.fc(embed[:,:int(embed.shape[1]/2)]), embed

    def sens_pred(self, embed):
        sens = self.sens_fc(embed[:,int(embed.shape[1]/2):])
        return sens
    
    def loss(self, out, batch, mode, retrain, epoch, dist_mode, indices_num):
        pred, embed = out if retrain else (out, None)
        labels, mask = batch['y'], batch[f'{mode}_mask']
        loss_pred = self.calculate_prediction_loss(pred, labels, mask)
        if retrain & (epoch % self.tepoch==0):
            indices1, indices2 = self.find_counterfactuals(embed, pred, batch['sens'], indices_num)
            causal_loss, stype_loss = self.calculate_causal_loss(embed, indices1, indices2, dist_mode)

    def calculate_prediction_loss(pred, labels, mask):
        if pred.dim() == 1:
            return F.binary_cross_entropy_with_logits(pred[mask], labels[mask].float())
        elif pred.dim() == 2:
            return F.cross_entropy(pred[mask], labels[mask])
        
    def calculate_causal_loss(self, embed, indices1, indices2, dist_mode):

    def loss(self, out, batch, mode, retrain, epoch, dist_mode):
        if retrain == False:
            preds = out
            labels, mask = batch['y'], batch[f'{mode}_mask']
            if preds.dim() == 1:
                return F.binary_cross_entropy_with_logits(preds[mask], labels[mask].float())
            elif preds.dim() == 2:
                return F.cross_entropy(preds[mask], labels[mask])
        elif retrain == True:
            preds, embed = out
            labels, mask = batch['y'], batch[f'{mode}_mask']
            if preds.dim() == 1:
                loss_pred = F.binary_cross_entropy_with_logits(preds[mask], labels[mask].float()) + F.binary_cross_entropy_with_logits(self.sens_pred(embed)[mask], batch['sens'][mask].float())
            elif preds.dim() == 2:
                loss_pred = F.cross_entropy(preds[mask], labels[mask]) + F.binary_cross_entropy_with_logits(self.sens_pred(embed)[mask], batch['sens'][mask].float())
            if epoch % self.tepoch == 0:
                indices1, indices2 = self.find_counterfactuals(embed, preds, batch['sens'], indices_num)
            half_dim = int(embed.shape[1]/2)
            causal_embed, style_embed = embed[:, :half_dim], embed[:, half_dim:]
            for i in range(indices_num):
                if causal_mode == 'MSE':    
                    causal_loss +=  MSE(causal_embed, causal_embed[indices1[:,i], :])
                    style_loss +=  MSE(style_embed, style_embed[indices2[:,i], :])
                elif causal_mode == 'train':
                    causal_loss += MSE(causal_embed[idx_train, :], causal_embed[indices1[:,i], :][idx_train, :])
                    style_loss += MSE(causal_embed[idx_train, :], causal_embed[indices1[:,i], :][idx_train, :])
                elif causal_mode == 'L1':
                    causal_loss +=  torch.mean(torch.abs(causal_embed - causal_embed[indices1[:,i], :]))
                    style_loss +=  torch.mean(torch.abs(style_embed - style_embed[indices2[:,i], :]))
                elif causal_mode == 'L2':
                    causal_loss +=  torch.mean((causal_embed - causal_embed[indices1[:,i], :])**2)
                    style_loss +=  torch.mean((style_embed - style_embed[indices2[:,i], :])**2)
                elif causal_mode == 'L1+L2':
                    causal_loss +=  torch.mean(torch.abs(causal_embed - causal_embed[indices1[:,i], :])) + torch.mean((causal_embed - causal_embed[indices1[:,i], :])**2)
                    style_loss +=  torch.mean(torch.abs(style_embed - style_embed[indices2[:,i], :])) + torch.mean((style_embed - style_embed[indices2[:,i], :])**2)
                elif causal_mode == 'contrastive':
                    causal_loss +=  contrastive_loss(causal_embed, causal_embed[indices1[:,i], :], causal_embed[indices2[:,i], :])
                    style_loss +=  contrastive_loss(style_embed, style_embed[indices2[:,i], :], style_embed[indices1[:,i], :]) 
                elif causal_mode == 'abscosine':
                    causal_loss +=  1-F.cosine_similarity(causal_embed, causal_embed[indices1[:,i], :]).mean()
                    style_loss +=  1-F.cosine_similarity(style_embed, style_embed[indices2[:,i], :]).mean()
                elif causal_mode == 'similarity':
                    causal_loss +=  similarity_loss(causal_embed, causal_embed[indices1[:,i], :], causal_embed[indices2[:,i], :])
                    style_loss +=  similarity_loss(style_embed, style_embed[indices2[:,i], :], style_embed[indices1[:,i], :])
                elif causal_mode == 'triplet':
                    causal_loss +=  triplet_loss(causal_embed, causal_embed[indices1[:,i], :], causal_embed[indices2[:,i], :])
                    style_loss +=  triplet_loss(style_embed, style_embed[indices2[:,i], :], style_embed[indices1[:,i], :])
            
            disentangle_loss = torch.abs(F.cosine_similarity(causal_embed, style_embed)).mean()
            decoder = InnerProductDecoder()
            pos_loss = -torch.log(decoder(embed, edge_index) + 1e-15).mean()
            neg_edge_index = negative_sampling(edge_index, num_nodes=adj.shape[0], num_neg_samples=edge_index.shape[1])
            neg_loss = -torch.log(1 - decoder(embed, neg_edge_index) + 1e-15).mean()
            rec_loss = pos_loss + neg_loss

            sens_coeff = 0
            sens_output = model.sens_pred(embed)
            sens_preds = (sens_output.squeeze()>0)
            sens_loss = F.binary_cross_entropy_with_logits(sens_output, sens.unsqueeze(1).float().to(device))
            # total loss
            loss_train =  loss_pred + args.causal_coeff*(causal_loss + style_loss) + args.disentangle_coeff*disentangle_loss + args.rec_coeff*rec_loss + sens_coeff*sens_loss
            return loss_train

    


    def find_counterfactuals(self, embed, preds, sens, indices_num):
        embed = embed.detach()
    
        # Compute label_pair and sens_pair
        pseudo_label = preds.reshape(-1, 1).bool()
        sens_torch = sens.reshape(-1, 1).bool()
        label_pair = pseudo_label.eq(pseudo_label.t())
        sens_pair = sens_torch.eq(sens_torch.t())

        # Compute select_adj and select_adj2
        select_adj = ~label_pair & sens_pair
        select_adj2 = label_pair & ~sens_pair

        # Compute the normalized distance
        distance = torch.cdist(embed, embed, p=2)
        distance = (distance - distance.min()) / (distance.max() - distance.min())

        # Find indices
        indices1 = self.find_indices(distance.cpu(), select_adj, indices_num)
        indices2 = self.find_indices(distance.cpu(), select_adj2, indices_num)
        
        return indices1, indices2


    def find_indices(self, distance, select_adj, num_indices):
        distance = distance.clone()
        distance[select_adj] = float('inf')
        _, indices = torch.topk(distance, num_indices, largest=False)
        return indices
