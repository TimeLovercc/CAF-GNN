import torch
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data
import os
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import distance_matrix
from torch_geometric.utils import dropout_adj, convert
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import Planetoid


class Credit(data.Dataset):
    def __init__(self, root=r'../data/credit',
                 dataset = "credit",
                 sens_attr = "Age",
                 sens_idx = 1,
                 predict_attr = "NoDefaultNextMonth",
                 transform = None,
                 label_number = 6000,
                 train = False):
        self.__dict__.update(locals())
        self.check_files()

    def check_files(self):
        adj, features, labels, idx_train, idx_val, idx_test, sens = self.load_credit(self.dataset, self.sens_attr,
                                                                                self.predict_attr, path=self.root,
                                                                                label_number=self.label_number,
                                                                                )
        edge_index = convert.from_scipy_sparse_matrix(adj)[0]
        features = self.transform(features) if self.transform is not None else features
        self.data = Data(x=features, edge_index=edge_index, y=labels, train_mask=idx_train, val_mask=idx_val, test_mask=idx_test, sens=sens)
        self.dataset = [self.data]
        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.data.to_dict()
 
    def load_credit(self, dataset, sens_attr="Age", predict_attr="NoDefaultNextMonth", path="../dataset/credit/", label_number=6000):
        # print('Loading {} dataset from {}'.format(dataset, path))
        idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove('Single')

        if os.path.exists(f'{path}/{dataset}_edges.txt'):
            edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
        else:
            edges_unordered = self.build_relationship(idx_features_labels[header], thresh=0.7)
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

    def build_relationship(self, x, thresh=0.25):
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

