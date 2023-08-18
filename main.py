#%%
import time
import argparse
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

from utils import *
from models import *
from torch_geometric.nn.models import InnerProductDecoder
from torch_geometric.utils import negative_sampling
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import dropout_adj, convert
from aif360.sklearn.metrics import consistency_score as cs
from aif360.sklearn.metrics import generalized_entropy_error as gee


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--sim_coeff', type=float, default=0.5,
                    help='regularization coeff for the self-supervised task')
parser.add_argument('--dataset', type=str, default='german',
                    choices=['nba','bail','loan', 'credit', 'german', 'pokec_z', 'pokec_n'])
parser.add_argument("--num_layers", type=int, default=1,
                        help="number of hidden layers")
parser.add_argument('--model', type=str, default='gcn',
                    choices=['gcn', 'sage', 'gin', 'jk', 'infomax', 'ssf', 'rogcn', 'our', 'our2'])
parser.add_argument('--encoder', type=str, default='gcn')
parser.add_argument('--tepoch', type=int, default=10)
parser.add_argument('--causal_coeff', type=float, default=10)
parser.add_argument('--disentangle_coeff', type=float, default=0)
parser.add_argument('--rec_coeff', type=float, default=0)
parser.add_argument('--cf_num', type=int, default=1)

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
print(args.dataset)

# Load credit_scoring dataset
if args.dataset == 'credit':
	sens_attr = "Age"  # column number after feature process is 1
	sens_idx = 1
	predict_attr = 'NoDefaultNextMonth'
	label_number = 6000
	path_credit = "./dataset/credit"
	adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(args.dataset, sens_attr,
	                                                                        predict_attr, path=path_credit,
	                                                                        label_number=label_number
	                                                                        )
	norm_features = feature_norm(features)
	norm_features[:, sens_idx] = features[:, sens_idx]
	features = norm_features

# Load german dataset
elif args.dataset == 'german':
	sens_attr = "Gender"  # column number after feature process is 0
	sens_idx = 0
	predict_attr = "GoodCustomer"
	label_number = 100
	path_german = "./dataset/german"
	adj, features, labels, idx_train, idx_val, idx_test, sens = load_german(args.dataset, sens_attr,
	                                                                        predict_attr, path=path_german,
	                                                                        label_number=label_number,
	                                                                        )
# Load bail dataset
elif args.dataset == 'bail':
	sens_attr = "WHITE"  # column number after feature process is 0
	sens_idx = 0
	predict_attr = "RECID"
	label_number = 100
	path_bail = "./dataset/bail"
	adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail(args.dataset, sens_attr, 
																			predict_attr, path=path_bail,
	                                                                        label_number=label_number,
	                                                                        )
	norm_features = feature_norm(features)
	norm_features[:, sens_idx] = features[:, sens_idx]
	features = norm_features

elif args.dataset == 'nba':
    sens_attr = "country"
    sens_idx = 35
    predict_attr = "SALARY"
    label_number = 100000
    path_nba = "./dataset/NBA"
    adj, features, labels, idx_train, idx_val, idx_test,sens = load_nba(args.dataset,
                                                                        sens_attr,
                                                                        predict_attr,
                                                                        path=path_nba,
                                                                        label_number=label_number,
                                                                        )
    norm_features = feature_norm(features)
    norm_features[:, sens_idx] = features[:, sens_idx]
    features = norm_features

elif args.dataset in ['pokec_z', 'pokec_n']:
    if args.dataset == 'pokec_z':       
        dataset = 'region_job'
    else:
        dataset = 'region_job_2'
    sens_attr = "region"
    sens_idx = 3
    predict_attr = "I_am_working_in_field"
    label_number = 100000
    seed = 20
    path_pokec="./dataset/pokec/"
    test_idx=False
    adj, features, labels, idx_train, idx_val, idx_test,sens = load_pokec(dataset,
                                                                        sens_attr,
                                                                        predict_attr,
                                                                        path=path_pokec,
                                                                        label_number=label_number,
                                                                        )
    norm_features = feature_norm(features)
    norm_features[:, sens_idx] = features[:, sens_idx]
    features = norm_features                                 

else:
	print('Invalid dataset name!!')
	exit(0)

edge_index = convert.from_scipy_sparse_matrix(adj)[0]

#%%    
# Model and optimizer
# num_class = labels.unique().shape[0]-1
num_class = 1
if args.model == 'gcn':
	model = GCN(nfeat=features.shape[1],
	            nhid=args.hidden,
	            nclass=num_class,
	            dropout=args.dropout)
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	model = model.to(device)
        
elif args.model == 'our':
    model = Our(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=num_class,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(device)

elif args.model == 'our2':
    model = Our_SAGE(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=num_class,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(device)

    
# Train model
t_total = time.time()
best_loss = 10000000
best_acc = 0
best_fair = 100
best_overall = 0
best_epoch = 0
features = features.to(device)
edge_index = edge_index.to(device)
labels = labels.to(device)

if args.model in ['our', 'our2']:
    model.load_state_dict(torch.load(f'./weights/weights_{args.model}_{args.dataset}_pretrain.pt'))

if args.model == 'rogcn':
    model.fit(features, adj, labels, idx_train, idx_val=idx_val, idx_test=idx_test, verbose=True, attention=False, train_iters=args.epochs)

for epoch in range(args.epochs+1):
    # t = time.time()
    if args.model in ['our', 'our2']:
        model.train()
        optimizer.zero_grad()
        output, embed = model(features, edge_index)

        # Binary Cross-Entropy  
        preds = (output.squeeze()>0).type_as(labels)
        loss_pred = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float().to(device))

        # find counterfactual
        if epoch % args.tepoch == 0:
            indices1, indices2 = find_counterfactual(embed, preds, sens, device)
        
        # counterfactual loss
        causal_dim = int(args.hidden/2)
        MSE = nn.MSELoss()
        causal_embed = embed[:, :causal_dim]
        causal_counterfactual = []
        causal_loss = 0
        style_embed = embed[:, causal_dim:]
        style_counterfactual = []
        style_loss = 0
        causal_mode = 'abscosine'
        for i in range(args.cf_num):
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
    
        # disentangle loss
        disentangle_loss = torch.abs(F.cosine_similarity(causal_embed, style_embed).mean())
        # ((torch.abs(F.cosine_similarity(causal_embed, style_embed))).sum()) / causal_embed.shape[0]
        # disentangle_loss = torch.mean(torch.log(1+torch.exp(+torch.matmul(causal_embed, style_embed.T))))
        # reconstruction loss
        decoder = InnerProductDecoder()
        pos_loss = -torch.log(decoder(embed, edge_index) + 1e-15).mean()
        neg_edge_index = negative_sampling(edge_index, num_nodes=adj.shape[0], num_neg_samples=edge_index.shape[1])
        neg_loss = -torch.log(1 - decoder(embed, neg_edge_index) + 1e-15).mean()
        rec_loss = pos_loss + neg_loss
        # sensitivity loss
        sens_coeff = 0
        sens_output = model.sens_pred(embed)
        sens_preds = (sens_output.squeeze()>0)
        sens_loss = F.binary_cross_entropy_with_logits(sens_output, sens.unsqueeze(1).float().to(device))
        # total loss
        loss_train =  loss_pred + args.causal_coeff*(causal_loss + style_loss) + args.disentangle_coeff*disentangle_loss + args.rec_coeff*rec_loss + sens_coeff*sens_loss

        auc_roc_train = roc_auc_score(labels.cpu().numpy()[idx_train], output.detach().cpu().numpy()[idx_train])
        loss_train.backward()
        optimizer.step()

        # Evaluate validation set performance separately,
        model.eval()
        output, _ = model(features, edge_index)

        # Binary Cross-Entropy
        preds = (output.squeeze()>0).type_as(labels)
        loss_val_pred = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float().to(device))
        
        # Total loss
        loss_val = loss_val_pred + args.causal_coeff*causal_loss + args.disentangle_coeff*disentangle_loss + args.rec_coeff*rec_loss 

        acc_val = accuracy(output[idx_val], labels[idx_val])
        auc_roc_val = roc_auc_score(labels.cpu().numpy()[idx_val], output.detach().cpu().numpy()[idx_val])
        f1_val = f1_score(labels[idx_val].cpu().numpy(), preds[idx_val].cpu().numpy())
        parity_val, equality_val = fair_metric(preds[idx_val].detach().cpu().numpy(), labels[idx_val].cpu().numpy(), sens[idx_val].numpy())
        parity_test, equality_test = fair_metric(preds[idx_test].detach().cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())

        if epoch % 10 == 0:
            print(f"[Train] Epoch {epoch}: train_loss: {loss_train.item():.4f} | train_auc_roc: {auc_roc_train:.4f} | val_loss: {loss_val.item():.4f} | val_auc_roc: {auc_roc_val:.4f}")
            print(f"[Val] Epoch {epoch}: causal_loss: {causal_loss.item():.4f} | disentangle_loss: {disentangle_loss.item():.4f} | rec_loss: {rec_loss.item():.4f}")
            print(f"[Val] Epoch {epoch}: acc: {acc_val:.4f} | f1: {f1_val:.4f} | parity: {parity_val:.4f} | equality: {equality_val:.4f}")
            print(f"[Test] Epoch {epoch}: parity: {parity_test:.4f} | equality: {equality_test:.4f}")

        # metric_val = loss_val + parity_val + equality_val
        # if metric_val < best_loss:
        if (loss_val < best_loss) and (epoch > 20) and (parity_val + equality_val < best_fair):
            best_loss = loss_val
            best_fair = parity_val + equality_val
            torch.save(model.state_dict(), f'./weights/weights_{args.model}_{args.dataset}_{args.seed}.pt')
            print(f"Saved model in Epoch {epoch}: Best overall metric: {best_loss:.4f}")
    # compute the time
    # t_total = time.time() - t
        


# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

if args.model in ['gcn', 'sage', 'gin', 'jk', 'infomax']:
    model.load_state_dict(torch.load(f'./weights/weights_{args.model}_{args.dataset}_{args.seed}.pt'))
    model.eval()
    output = model(features.to(device), edge_index.to(device))
    counter_features = features.clone()
    counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
    counter_output = model(counter_features.to(device), edge_index.to(device))


elif args.model == 'rogcn':
    model.load_state_dict(torch.load(f'./weights/weights_{args.model}_{args.dataset}_{args.seed}.pt'))
    model.eval()
    model = model.to('cpu')
    output = model.predict(features.to('cpu'))
    counter_features = features.to('cpu').clone()
    counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
    counter_output = model.predict(counter_features.to('cpu'))

elif args.model in ['our', 'our2']:
    model.load_state_dict(torch.load(f'./weights/weights_{args.model}_{args.dataset}_{args.seed}.pt'))
    model.eval()
    output, _ = model(features.to(device), edge_index.to(device))
    counter_features = features.clone()
    counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
    counter_output, _ = model(counter_features.to(device), edge_index.to(device))

else:
    model.load_state_dict(torch.load(f'./weights/weights_{args.model}_{args.dataset}_{args.seed}.pt'))
    model.eval()
    emb = model(features.to(device), edge_index.to(device))
    output = model.predict(emb)
    counter_features = features.clone()
    counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
    counter_output = model.predict(model(counter_features.to(device), edge_index.to(device)))

# Report
output_preds = (output.squeeze()>0).type_as(labels)
counter_output_preds = (counter_output.squeeze()>0).type_as(labels)
auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()])
counterfactual_fairness = 1 - (output_preds.eq(counter_output_preds)[idx_test].sum().item()/idx_test.shape[0])

parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())
f1_s = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())
acc_s = accuracy(output[idx_test], labels[idx_test])
# print(output_preds[idx_test].cpu().numpy())
# draw_ROC(output.detach().cpu().numpy()[idx_test.cpu()], labels.cpu().numpy()[idx_test.cpu()])

# print report
print(f"The AUCROC of estimator: {auc_roc_test:.4f} | F1-score: {f1_s} | Accuracy: {acc_s}")
print(f'Parity: {parity} | Equality: {equality} | Counterfactual Fairness: {counterfactual_fairness}')

with open(f'./reports/{args.dataset}_report.txt', 'a+') as f:
    f.write(f'{auc_roc_test:.4f} | {f1_s:.4f} | {acc_s:.4f} | {parity:.4f} | {equality:.4f} |  {counterfactual_fairness:.4f} || {args.model} | {args.lr} | {args.seed} | {args.hidden} | {args.weight_decay} || {args.causal_coeff} | {args.disentangle_coeff} | {args.rec_coeff} | {args.cf_num}\n')
