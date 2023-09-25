import os
import dgl
import torch
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import distance_matrix
import torch.nn.functional as F
from sklearn import metrics
import matplotlib.pyplot as plt
import copy
import networkx as nx
import scipy.io as scio


def draw_ROC(pred, labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('ROC2.png')

def contrastive_loss(embed, pos, neg, margin=1):
    pos_dist = (embed - pos).pow(2).sum(1)
    neg_dist = (embed - neg).pow(2).sum(1)
    basic_loss = pos_dist - neg_dist + margin
    loss = torch.mean(torch.max(basic_loss, torch.zeros_like(basic_loss)))
    return loss

def similarity_loss(embed, pos, neg, margin=1):
    pos_dist = F.cosine_similarity(embed, pos, dim=1)
    neg_dist = F.cosine_similarity(embed, neg, dim=1)
    basic_loss = neg_dist - pos_dist + margin
    loss = torch.mean(torch.max(basic_loss, torch.zeros_like(basic_loss)))
    return loss

def triplet_loss(embed, pos, neg, margin=1):
    pos_dist = (embed - pos).pow(2).sum(1)
    neg_dist = (embed - neg).pow(2).sum(1)
    basic_loss = pos_dist - neg_dist + margin
    loss = torch.mean(torch.max(basic_loss, torch.zeros_like(basic_loss)))
    return loss


def find_counterfactual(embed, preds, sens, device):
    # compute the time
    # print(f'current time: {start_time}')
    # find counterfactual examples
    embed = embed.detach()
    pseudo_label = preds.reshape(-1, 1).bool()
    sens_torch = sens.reshape(-1, 1).bool().to(device)
    # and operation
    label_pair = torch.logical_or((pseudo_label & pseudo_label.T), (~pseudo_label & ~pseudo_label.T))
    sens_pair = torch.logical_or((sens_torch & sens_torch.T), (~sens_torch & ~sens_torch.T))
    # delele label
    del pseudo_label
    del sens_torch
    torch.cuda.empty_cache()
    # compute select
    select_adj = ~(label_pair & (~sens_pair))
    select_adj2 = ~((~label_pair) & sens_pair)
    # release label_pair and sens_pair
    del label_pair
    del sens_pair
    torch.cuda.empty_cache()
    # compute the distance
    distance = torch.cdist(embed, embed, p=2)
    distance -= distance.min()
    distance /= distance.max()
    total_distance = distance.cpu()
    # label same, sens different, same is 1, different is 0
    total_distance[select_adj] = 100
    _, indices1 = torch.topk(total_distance, 100, largest = False)
    # indices1 = indices1[:, 100]
    # delete total_distance
    del total_distance
    del distance
    torch.cuda.empty_cache()
    # compute the distance
    distance = torch.cdist(embed, embed, p=2)
    distance -= distance.min()
    distance /= distance.max()
    total_distance2 = distance.cpu()
    # label different, sens same, same is 1, different is 0
    total_distance2[select_adj2] = 100
    _, indices2 = torch.topk(total_distance2, 100, largest = False)
    # indices2 = indices2[:, 100]
    # delete total_distance2
    del total_distance2
    del distance
    del select_adj
    del select_adj2
    del _
    torch.cuda.empty_cache()
    # compute the time
    # end_time = time.time()
    # print(f'current time: {end_time}')
    # print("Time: ", end_time - start_time)
    return indices1, indices2

def find_counterfactual_new(embed, preds, sens, device):
    # compute the time
    import time
    start_time = time.time()
    # print(f'current time: {start_time}')
    # find counterfactual examples
    embed = embed.detach()
    pseudo_label = preds.reshape(-1, 1).bool()
    sens_torch = sens.reshape(-1, 1).bool().to(device)
    distance = torch.cdist(embed, embed, p=2)
    distance -= distance.min()
    distance /= distance.max()
    torch.save(distance, 'distance_tmp.pt')
    del distance
    torch.cuda.empty_cache()
    # and operation
    label_pair = torch.logical_or((pseudo_label & pseudo_label.T), (~pseudo_label & ~pseudo_label.T))
    sens_pair = torch.logical_or((sens_torch & sens_torch.T), (~sens_torch & ~sens_torch.T))
    select_adj = ~(label_pair & (~sens_pair))
    select_adj2 = ~((~label_pair) & sens_pair)
    # release label_pair and sens_pair
    del label_pair
    del sens_pair
    torch.cuda.empty_cache()
    total_distance = torch.load('distance_tmp.pt').to(device)
    # label same, sens different, same is 1, different is 0
    total_distance[select_adj] = 100
    _, indices1 = torch.topk(total_distance, 100, largest = False)
    # indices1 = indices1[:, 100]
    # delete total_distance
    del total_distance
    del select_adj
    torch.cuda.empty_cache()
    total_distance2 = torch.load('distance_tmp.pt').to(device)
    # label different, sens same, same is 1, different is 0
    total_distance2[select_adj2] = 100
    _, indices2 = torch.topk(total_distance2, 100, largest = False)
    # indices2 = indices2[:, 100]
    # delete total_distance2
    del total_distance2
    del select_adj2
    torch.cuda.empty_cache()
    # compute the time
    # end_time = time.time()
    # print(f'current time: {end_time}')
    # print("Time: ", end_time - start_time)
    return indices1, indices2

def find_counterfactual_old2(embed, preds, sens, device):
    # compute the time
    import time
    start_time = time.time()
    # print(f'current time: {start_time}')
    # find counterfactual examples
    embed = embed.detach()
    pseudo_label = preds.reshape(-1, 1).bool()
    sens_torch = sens.reshape(-1, 1).bool().to(device)
    distance = torch.cdist(embed, embed, p=2)
    distance -= distance.min()
    distance /= distance.max()
    # and operation
    label_pair = torch.logical_or((pseudo_label & pseudo_label.T), (~pseudo_label & ~pseudo_label.T))
    sens_pair = torch.logical_or((sens_torch & sens_torch.T), (~sens_torch & ~sens_torch.T))
    select_adj = ~(label_pair & (~sens_pair))
    select_adj2 = ~((~label_pair) & sens_pair)
    # release label_pair and sens_pair
    del label_pair
    del sens_pair
    torch.cuda.empty_cache()
    total_distance = distance
    total_distance2 = copy.deepcopy(distance)
    # label same, sens different, same is 1, different is 0
    total_distance[select_adj] = 100
    _, indices1 = torch.topk(total_distance, 100, largest = False)
    # indices1 = indices1[:, 100]
    # delete total_distance
    del total_distance
    torch.cuda.empty_cache()
    # label different, sens same, same is 1, different is 0
    total_distance2[select_adj2] = 100
    _, indices2 = torch.topk(total_distance2, 100, largest = False)
    # indices2 = indices2[:, 100]
    # delete total_distance2
    del total_distance2
    torch.cuda.empty_cache()
    # compute the time
    # end_time = time.time()
    # print(f'current time: {end_time}')
    # print("Time: ", end_time - start_time)
    return indices1, indices2


def find_counterfactual_old(embed, preds, sens, device):
    # find counterfactual examples
    embed = embed.detach()
    pseudo_label = preds.reshape(-1, 1).float()
    sens_torch = sens.reshape(-1, 1).float().to(device)
    causal_dim = int(embed.shape[1]/2)
    causal_embed = embed[:, :causal_dim]
    style_embed = embed[:, causal_dim:]
    causal_distance = torch.cdist(causal_embed, causal_embed, p=2)
    style_distance = torch.cdist(style_embed, style_embed, p=2)
    causal_distance -= causal_distance.min()
    causal_distance /= causal_distance.max()
    style_distance -= style_distance.min()
    style_distance /= style_distance.max()
    label_pair = 1 - torch.cdist(pseudo_label, pseudo_label, p=1)
    sens_pair = 1 - torch.cdist(sens_torch, sens_torch, p=1)
    # Weighted for causal and style importance
    alpha = 1
    beta = 1
    total_distance = + alpha * causal_distance + beta * style_distance
    total_distance2 = + alpha * causal_distance + beta * style_distance
    # label same, sens different, same is 1, different is 0
    select_adj = label_pair * (1 - sens_pair)
    select_adj = (select_adj == 1)
    total_distance[~select_adj] = 100
    _, indices1 = torch.topk(total_distance, 100, largest = False)
    # indices1 = indices1[:, 100]
    torch.cuda.empty_cache()
    # label different, sens same, same is 1, different is 0
    select_adj2 = (1 - label_pair) * sens_pair
    select_adj2 = (select_adj2 == 1)
    total_distance2[~select_adj2] = 100
    _, indices2 = torch.topk(total_distance2, 100, largest = False)
    # indices2 = indices2[:, 100]
    torch.cuda.empty_cache()
    return indices1, indices2


def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()


def ssf_validation(model, x_1, edge_index_1, x_2, edge_index_2, y, idx_val, args, device):
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    # projector
    p1 = model.projection(z1)
    p2 = model.projection(z2)

    # predictor
    h1 = model.prediction(p1)
    h2 = model.prediction(p2)

    l1 = model.D(h1[idx_val], p2[idx_val])/2
    l2 = model.D(h2[idx_val], p1[idx_val])/2
    sim_loss = args.sim_coeff*(l1+l2)

    # classifier
    c1 = model.classifier(z1)
    c2 = model.classifier(z2)

    # Binary Cross-Entropy
    l3 = F.binary_cross_entropy_with_logits(c1[idx_val], y[idx_val].unsqueeze(1).float().to(device))/2
    l4 = F.binary_cross_entropy_with_logits(c2[idx_val], y[idx_val].unsqueeze(1).float().to(device))/2

    return sim_loss, l3+l4

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh*max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    # print('building edge relationship complete')
    idx_map =  np.array(idx_map)
    
    return idx_map


def load_synthetic(dataset, label_number = 500):
    # path = './dataset/synthetic/synthetic_2000_50_0.4_0.3_0.01_0.01_0.6.mat'
    path = './dataset/synthetic/synthetic_2000_50_0.4_0.3_0.01_0.01_0.6_32.mat'
    data = scio.loadmat(path)
    features = data['x']
    adj = data['adj']
    labels = data['labels'][0]
    sens = data['sens'][0]
    features = np.concatenate([sens.reshape(-1,1), features], axis=1)

    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # load counterfactual data
    features_cf = data['x_cf']
    adj_cf = data['adj_cf']
    adj_cf = adj_cf + adj_cf.T.multiply(adj_cf.T > adj_cf) - adj_cf.multiply(adj_cf.T > adj_cf)
    adj_cf = adj_cf + sp.eye(adj_cf.shape[0])
    features_cf = np.concatenate([1-sens.reshape(-1,1), features_cf], axis=1)
    features_cf = torch.FloatTensor(features_cf)


    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                            label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test, sens, adj_cf, features_cf

def load_credit(dataset, sens_attr="Age", predict_attr="NoDefaultNextMonth", path="./dataset/credit/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('Single')

#    # Normalize MaxBillAmountOverLast6Months
#    idx_features_labels['MaxBillAmountOverLast6Months'] = (idx_features_labels['MaxBillAmountOverLast6Months']-idx_features_labels['MaxBillAmountOverLast6Months'].mean())/idx_features_labels['MaxBillAmountOverLast6Months'].std()
#
#    # Normalize MaxPaymentAmountOverLast6Months
#    idx_features_labels['MaxPaymentAmountOverLast6Months'] = (idx_features_labels['MaxPaymentAmountOverLast6Months'] - idx_features_labels['MaxPaymentAmountOverLast6Months'].mean())/idx_features_labels['MaxPaymentAmountOverLast6Months'].std()
#
#    # Normalize MostRecentBillAmount
#    idx_features_labels['MostRecentBillAmount'] = (idx_features_labels['MostRecentBillAmount']-idx_features_labels['MostRecentBillAmount'].mean())/idx_features_labels['MostRecentBillAmount'].std()
#
#    # Normalize MostRecentPaymentAmount
#    idx_features_labels['MostRecentPaymentAmount'] = (idx_features_labels['MostRecentPaymentAmount']-idx_features_labels['MostRecentPaymentAmount'].mean())/idx_features_labels['MostRecentPaymentAmount'].std()
#
#    # Normalize TotalMonthsOverdue
#    idx_features_labels['TotalMonthsOverdue'] = (idx_features_labels['TotalMonthsOverdue']-idx_features_labels['TotalMonthsOverdue'].mean())/idx_features_labels['TotalMonthsOverdue'].std()

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_bail(dataset, sens_attr="WHITE", predict_attr="RECID", path="../dataset/bail/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    
    # # Normalize School
    # idx_features_labels['SCHOOL'] = 2*(idx_features_labels['SCHOOL']-idx_features_labels['SCHOOL'].min()).div(idx_features_labels['SCHOOL'].max() - idx_features_labels['SCHOOL'].min()) - 1

    # # Normalize RULE
    # idx_features_labels['RULE'] = 2*(idx_features_labels['RULE']-idx_features_labels['RULE'].min()).div(idx_features_labels['RULE'].max() - idx_features_labels['RULE'].min()) - 1

    # # Normalize AGE
    # idx_features_labels['AGE'] = 2*(idx_features_labels['AGE']-idx_features_labels['AGE'].min()).div(idx_features_labels['AGE'].max() - idx_features_labels['AGE'].min()) - 1

    # # Normalize TSERVD
    # idx_features_labels['TSERVD'] = 2*(idx_features_labels['TSERVD']-idx_features_labels['TSERVD'].min()).div(idx_features_labels['TSERVD'].max() - idx_features_labels['TSERVD'].min()) - 1

    # # Normalize FOLLOW
    # idx_features_labels['FOLLOW'] = 2*(idx_features_labels['FOLLOW']-idx_features_labels['FOLLOW'].min()).div(idx_features_labels['FOLLOW'].max() - idx_features_labels['FOLLOW'].min()) - 1

    # # Normalize TIME
    # idx_features_labels['TIME'] = 2*(idx_features_labels['TIME']-idx_features_labels['TIME'].min()).div(idx_features_labels['TIME'].max() - idx_features_labels['TIME'].min()) - 1

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_german(dataset, sens_attr="Gender", predict_attr="GoodCustomer", path="../dataset/german/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')

    # Sensitive Attribute
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

#    for i in range(idx_features_labels['PurposeOfLoan'].unique().shape[0]):
#        val = idx_features_labels['PurposeOfLoan'].unique()[i]
#        idx_features_labels['PurposeOfLoan'][idx_features_labels['PurposeOfLoan'] == val] = i

#    # Normalize LoanAmount
#    idx_features_labels['LoanAmount'] = 2*(idx_features_labels['LoanAmount']-idx_features_labels['LoanAmount'].min()).div(idx_features_labels['LoanAmount'].max() - idx_features_labels['LoanAmount'].min()) - 1
#
#    # Normalize Age
#    idx_features_labels['Age'] = 2*(idx_features_labels['Age']-idx_features_labels['Age'].min()).div(idx_features_labels['Age'].max() - idx_features_labels['Age'].min()) - 1
#
#    # Normalize LoanDuration
#    idx_features_labels['LoanDuration'] = 2*(idx_features_labels['LoanDuration']-idx_features_labels['LoanDuration'].min()).div(idx_features_labels['LoanDuration'].max() - idx_features_labels['LoanDuration'].min()) - 1
#
    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.8)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
   
    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_nba(dataset,sens_attr,predict_attr, path="../dataset/nba/", label_number=100):
    """
    from enyan
    """
    # print('Loading {} dataset from {}'.format(dataset,path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    # header.remove(sens_attr)
    header.remove(predict_attr)


    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    # labels[labels == -1] = 0
    

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path,"{}_relationship.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])


    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_pokec(dataset,sens_attr,predict_attr, path="../dataset/pokec/", label_number=500):
    # print('Loading {} dataset from {}'.format(dataset,path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    # header.remove(sens_attr)
    header.remove(predict_attr)


    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    # labels[labels == -1] = 0
    labels[labels >= 1] = 1
    

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path,"{}_relationship.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])


    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens

def load_pokec_old(dataset,sens_attr,predict_attr, path="../dataset/pokec/", label_number=1000):
    """Load data, this is from enyan's work"""
    print('Loading {} dataset from {}'.format(dataset,path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    # header.remove(sens_attr)
    header.remove(predict_attr)

    # Sensitive Attribute
    # idx_features_labels[sens_attr][idx_features_labels[sens_attr] == 'Female'] = 1
    # idx_features_labels[sens_attr][idx_featur es_labels[sens_attr] == 'Male'] = 0

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == 0] = 1
    labels[labels == -1] = 0
    

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path,"{}_relationship.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import random
    seed = 20
    random.seed(seed)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    # sens_idx = set(np.where(sens >= 0)[0])
    # idx_test = np.asarray(list(sens_idx & set(idx_test)))
    # idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    # random.seed(seed)
    # random.shuffle(idx_sens_train)
    # idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    

    # random.shuffle(sens_idx)

    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_pokec_new(dataset, sens_attr, predict_attr, path="../pokec_dataset/", label_number=1000,sens_number=500,seed=19,test_idx=False):
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    
    header = list(idx_features_labels.columns)
    header.remove("user_id")
    # header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    sens = idx_features_labels[sens_attr].values
    #Only nodes for which label and sensitive attributes are available are utilized 
    sens_idx = set(np.where(sens >= 0)[0])
    label_idx = np.where(labels >= 0)[0]
    idx_used = np.asarray(list(sens_idx & set(label_idx)))
    idx_nonused = np.asarray(list(set(np.arange(len(labels))).difference(set(idx_used))))

    features = features[idx_used, :]
    labels = labels[idx_used]
    sens = sens[idx_used]

    idx = np.array(idx_features_labels["user_id"], dtype=int)
    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=int)

    idx_n = idx[idx_nonused]
    idx = idx[idx_used]
    used_ind1 = [i for i, elem in enumerate(edges_unordered[:, 0]) if elem not in idx_n]
    used_ind2 = [i for i, elem in enumerate(edges_unordered[:, 1]) if elem not in idx_n]
    intersect_ind = list(set(used_ind1) & set(used_ind2))
    edges_unordered = edges_unordered[intersect_ind, :]
    # build graph

    idx_map = {j: i for i, j in enumerate(idx)}
    edges_un = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=int).reshape(edges_unordered.shape)

    
    adj = sp.coo_matrix((np.ones(edges_un.shape[0]), (edges_un[:, 0], edges_un[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    G = nx.from_scipy_sparse_matrix(adj)
    g_nx_ccs = (G.subgraph(c).copy() for c in nx.connected_components(G))
    g_nx = max(g_nx_ccs, key=len)

    import random
    random.seed(seed)
    node_ids = list(g_nx.nodes())
    idx_s=node_ids
    random.shuffle(idx_s)
    
    features=features[idx_s,:]
    features=features[:,np.where(np.std(np.array(features.todense()),axis=0)!=0)[0]] 
    
    # features=torch.FloatTensor(np.array(features.todense()))
    labels=torch.LongTensor(labels[idx_s])
    
    sens=torch.LongTensor(sens[idx_s])
    idx_map_n = {j: int(i) for i, j in enumerate(idx_s)}

    idx_nonused2 = np.asarray(list(set(np.arange(len(list(G.nodes())))).difference(set(idx_s))))
    used_ind1 = [i for i, elem in enumerate(edges_un[:, 0]) if elem not in idx_nonused2]
    used_ind2 = [i for i, elem in enumerate(edges_un[:, 1]) if elem not in idx_nonused2]
    intersect_ind = list(set(used_ind1) & set(used_ind2))
    edges_un = edges_un[intersect_ind, :]
    edges = np.array(list(map(idx_map_n.get, edges_un.flatten())),
                     dtype=int).reshape(edges_un.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import random
    random.seed(seed)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    # sens = idx_features_labels[sens_attr].values.astype(int)
    # sens = torch.FloatTensor(sens)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens




def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2*(features - min_values).div(max_values-min_values) - 1

def accuracy(output, labels):
    output = output.squeeze()
    preds = (output>0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_softmax(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
