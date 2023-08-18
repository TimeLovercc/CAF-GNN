#%%
import dgl
import ipdb
import time
import argparse
import numpy as np

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

from utils import *
from models import *
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import convert


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.003,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default='german',
                    choices=['nba','bail','loan', 'credit', 'german', 'pokec_z', 'pokec_n', 'synthetic'])
parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
parser.add_argument('--model', type=str, default='our2',
                    choices=['gcn', 'sage', 'gin', 'jk', 'infomax', 'ssf', 'rogcn', 'our', 'our2'])
parser.add_argument('--encoder', type=str, default='gcn')


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
# print(args.dataset)


if args.dataset == 'synthetic':
    sens_idx = 0
    label_number = 500
    adj, features, labels, idx_train, idx_val, idx_test, sens, adj_cf, features_cf = load_synthetic(args.dataset, label_number=label_number)
    norm_features = feature_norm(features)
    norm_features[:, sens_idx] = features[:, sens_idx]
    features = norm_features
    # counterfactual
    norm_features_cf = feature_norm(features_cf)
    norm_features_cf[:, sens_idx] = features_cf[:, sens_idx]
    features_cf = norm_features_cf
    edge_index_cf = convert.from_scipy_sparse_matrix(adj_cf)[0]

elif args.dataset == 'credit':
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
if args.model == 'our':
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
features = features.to(device)
edge_index = edge_index.to(device)
labels = labels.to(device)

# if args.model == 'our':
#     model.load_state_dict(torch.load(f'./weights/weights_{args.model}_{args.dataset}_{args.seed}_pretrain.pt'))


for epoch in range(args.epochs+1):
    t = time.time()
    
    if args.model in ['our', 'our2']:
        model.train()
        optimizer.zero_grad()
        output, embed = model(features, edge_index)

        # Binary Cross-Entropy  
        preds = (output.squeeze()>0).type_as(labels)
        loss_pred = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float().to(device))

        # Fairness loss
        # sens_output = model.sens_pred(embed)
        # sens_preds = (sens_output.squeeze()>0)
        # sens_loss = F.binary_cross_entropy_with_logits(sens_output, sens.unsqueeze(1).float().to(device))

        # predict
        loss_train = loss_pred 

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
        loss_val = loss_val_pred 

        acc_val = accuracy(output[idx_val], labels[idx_val])
        auc_roc_val = roc_auc_score(labels.cpu().numpy()[idx_val], output.detach().cpu().numpy()[idx_val])
        f1_val = f1_score(labels[idx_val].cpu().numpy(), preds[idx_val].cpu().numpy())
        parity_val, equality_val = fair_metric(preds[idx_val].detach().cpu().numpy(), labels[idx_val].cpu().numpy(), sens[idx_val].numpy())

        if epoch % 10 == 0:
            print(f"[Train] Epoch {epoch}:train_loss: {loss_train.item():.4f} | train_auc_roc: {auc_roc_train:.4f} | val_loss: {loss_val.item():.4f} | val_auc_roc: {auc_roc_val:.4f}")
            print(f"[Val] Epoch {epoch}: acc: {acc_val:.4f} | f1: {f1_val:.4f} | parity: {parity_val:.4f} | equality: {equality_val:.4f}")

        if loss_val < best_loss and epoch > 20:
            best_loss = loss_val
            torch.save(model.state_dict(), f'./weights/weights_{args.model}_{args.dataset}_pretrain.pt')
            print(f"Saved model in Epoch {epoch}: Best loss val: {best_loss:.4f}")
        


# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


if args.model in ['our', 'our2']:
    model.load_state_dict(torch.load(f'./weights/weights_{args.model}_{args.dataset}_pretrain.pt'))
    model.eval()
    output, _ = model(features.to(device), edge_index.to(device))
    counter_features = features.clone()
    counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
    counter_output, _ = model(counter_features.to(device), edge_index.to(device))


# Report
output_preds = (output.squeeze()>0).type_as(labels)
counter_output_preds = (counter_output.squeeze()>0).type_as(labels)
auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()])
counterfactual_fairness = 1 - (output_preds.eq(counter_output_preds)[idx_test].sum().item()/idx_test.shape[0])

parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())
f1_s = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())
acc_s = accuracy(output[idx_test], labels[idx_test])

# print report
print(f"The AUCROC of estimator: {auc_roc_test:.4f} | F1-score: {f1_s} | Accuracy: {acc_s}")
print(f'Parity: {parity} | Equality: {equality} | Counterfactual Fairness: {counterfactual_fairness}')

with open(f'./reports/{args.dataset}_report.txt', 'a+') as f:
    f.write(f'Pretrain: {auc_roc_test:.4f} | {f1_s:.4f} | {acc_s:.4f} | {parity:.4f} | {equality:.4f} |  {counterfactual_fairness:.4f} || {args.model} | {args.lr} | {args.seed} | {args.hidden} | {args.weight_decay} \n')