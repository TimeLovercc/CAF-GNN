import torch.nn as nn
from models import *
import torch
import gc

def get_model(nfeat, args):
    if args.model == "gcn":
        model = GCN_Body(nfeat,args.num_hidden,args.dropout)
    elif args.model == "sage":
        model = SAGE_Body(nfeat,args.num_hidden,args.dropout)
    else:
        print("Model not implement")
        return

    return model

class FairGNN(nn.Module):

    def __init__(self, nfeat, args):
        super(FairGNN,self).__init__()

        nhid = args.num_hidden
        dropout = args.dropout
        self.estimator = GCN(nfeat,args.hidden,1,dropout)
        self.GNN = get_model(nfeat,args)
        self.classifier = nn.Linear(nhid,1)
        self.adv = nn.Linear(nhid,1)

        # G_params = list(self.GNN.parameters()) + list(self.classifier.parameters()) + list(self.estimator.parameters())
        # self.optimizer_G = torch.optim.Adam(G_params, lr = args.lr, weight_decay = args.weight_decay)
        # self.optimizer_A = torch.optim.Adam(self.adv.parameters(), lr = args.lr, weight_decay = args.weight_decay)

        self.args = args
        # self.criterion = nn.BCEWithLogitsLoss()

        self.G_loss = 0
        self.A_loss = 0

    def forward(self, x, edge_index):
        s = self.estimator(x, edge_index)
        z = self.GNN(x, edge_index)
        y = self.classifier(z)
        return y, s, z
    
    def optimize(self,g,x,labels,idx_train,sens,idx_sens_train):
        self.train()

        ### update E, G
        self.adv.requires_grad_(False)
        self.optimizer_G.zero_grad()

        s = self.estimator(g,x)
        h = self.GNN(g,x)
        y = self.classifier(h)

        s_g = self.adv(h)

        s_score = torch.sigmoid(s.detach())
        # s_score = (s_score > 0.5).float()
        s_score[idx_sens_train]=sens[idx_sens_train].unsqueeze(1).float()
        y_score = torch.sigmoid(y)
        self.cov =  torch.abs(torch.mean((s_score - torch.mean(s_score)) * (y_score - torch.mean(y_score))))
        
        self.cls_loss = self.criterion(y[idx_train],labels[idx_train].unsqueeze(1).float())
        self.adv_loss = self.criterion(s_g,s_score)
        
        self.G_loss = self.cls_loss  + self.args.alpha * self.cov - self.args.beta * self.adv_loss
        self.G_loss.backward()
        self.optimizer_G.step()

        ## update Adv
        self.adv.requires_grad_(True)
        self.optimizer_A.zero_grad()
        s_g = self.adv(h.detach())
        self.A_loss = self.criterion(s_g,s_score)
        self.A_loss.backward()
        self.optimizer_A.step()


