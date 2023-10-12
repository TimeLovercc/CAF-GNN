import re
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
import scipy.io as scio

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def find_latest_best_checkpoint(base_dir):
    # Recursively find all 'lightning_logs' directories
    all_runs = list(Path(base_dir).rglob('*'))  # retrieve all files and folders
    all_runs = [run for run in all_runs if run.is_dir() and 'lightning_logs' in str(run)]  # filter only relevant directories

    # Function to convert the directory name to a datetime object
    def get_datetime_from_dir(dir):
        dir_name = str(dir)  # convert Path object to string
        match = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', dir_name)
        if match:
            time_str = match.group(0)
            # Replace '_' with space and the first two '-' with ':'
            time_str = time_str.replace('_', ' ').replace('-', ':', 2)
            # Now replace the remaining '-' with ':'
            time_str = time_str.replace('-', ':')
            try:
                date_time_obj = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
                # Debug print to show the extracted datetime
                return date_time_obj
            except ValueError as e:
                # Debug print to show directories that failed the datetime extraction
                return datetime.min  # return the smallest possible datetime for directories that don't match
        else:
            # Debug print to show directories that failed the datetime extraction
            return datetime.min  # return the smallest possible datetime for directories that don't match


    # Sort the runs based on the datetime and get the latest
    latest_run = max(all_runs, key=lambda dir: get_datetime_from_dir(dir))

    # Find the 'best.ckpt' file in the latest run directory
    best_checkpoint = next(Path(latest_run).rglob('*best.ckpt'), None)

    if best_checkpoint:
        return best_checkpoint
    else:
        print(f"No best checkpoint found in the latest run at: {latest_run}")
        return None
    





def pre_analysis(adj, labels, sens):
    '''
    :param
        labels: n
        sens: n
        adj: csr_sparse, n x n
    :return: this function analyze:
        1. the correlation between the sensitive attributes of neighbors and the labels
        2. the correlation between the sensitive attributes of itself and the labels
        S_N(i), Y_i | S_i  not independent ->
    '''
    adj_noself = adj.copy()
    adj_noself.setdiag(0)  # remove edge v -> v
    if (adj_noself != adj_noself.T).nnz == 0:
        print("symmetric!")
    else:
        print("not symmetric!")
    adj_degree = adj.sum(axis=1)
    ave_degree = adj_degree.sum()/len(adj_degree)
    print('averaged degree: ', ave_degree, ' max degree: ', adj_degree.max(), ' min degree: ', adj_degree.min())

    # inter- and intra- connections
    node_num = adj.shape[0]
    edge_num = (len(adj.nonzero()[0]) - node_num) / 2
    intra_num = 0
    for u, v in zip(*adj.nonzero()):  # u -> v
        if u >= v:
            continue
        if sens[u] == sens[v]:
            intra_num += 1
    print("edge num: ", edge_num, " intra-group edge: ", intra_num, " inter-group edge: ", edge_num - intra_num)

    # row-normalize
    adj_noself = normalize(adj_noself, norm='l1', axis=1)
    nb_sens_ave = adj_noself @ sens  # n x 1, average sens of 1-hop neighbors

    # Y_i, S_i
    #pyplot.scatter(labels, sens)
    #pyplot.show()

    cov_results = stats_cov(labels, sens)
    print('correlation between Y and S:', cov_results)

    # S_N(i), Y_i | S_i
    cov_nb_results = stats_cov(labels, nb_sens_ave)
    print('correlation between Y and neighbors (not include self)\' S:', cov_nb_results)

    # R^2
    X = sens.reshape(node_num, -1)
    reg = LinearRegression().fit(X, labels)
    y_pred = reg.predict(X)
    R2 = r2_score(labels, y_pred)
    print('R2 - self: ', R2, ' ', reg.score(X, labels))

    X = nb_sens_ave.reshape(node_num, -1)
    reg = LinearRegression().fit(X, labels)
    y_pred = reg.predict(X)
    R2 = r2_score(labels, y_pred)
    print('R2 - neighbor: ', R2, ' ', reg.score(X, labels))

    return



def generate_synthetic_data_old(path):

    sens = np.random.binomial(n=1, p=p, size=n)
    embedding = np.random.normal(loc=0, scale=1, size=(n, z_dim))
    feat_idxs = random.sample(range(z_dim), dim)
    v = np.random.normal(0, 1, size=(1, dim))
    features = embedding[:, feat_idxs] + (np.dot(sens.reshape(-1,1), v))  # (n x dim) + (1 x dim) -> n x dim

    adj = np.zeros((n, n))
    sens_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):  # i<=j
            if i == j:
                sens_sim[i][j] = 1
                continue
            sens_sim[i][j] = sens_sim[j][i] = (sens[i] == sens[j])
            # sim_ij = 1 - spatial.distance.cosine(embedding[i], embedding[j])  # [-1, 1]
            # adj[i][j] = adj[j][i] = sim_ij + alpha * (sens[i] == sens[j])

    similarities = cosine_similarity(embedding)  # n x n
    adj = similarities + alpha * sens_sim

    print('adj max: ', adj.max(), ' min: ', adj.min())
    adj[np.where(adj >= 0.4)] = 1
    adj[np.where(adj < 0.4)] = 0
    adj = sparse.csr_matrix(adj)

    w = np.random.normal(0, 1, size=(z_dim, 1))
    w_s = 1

    adj_norm = normalize(adj, norm='l1', axis=1)
    nb_sens_ave = adj_norm @ sens  # n x 1, average sens of 1-hop neighbors

    dd = np.matmul(embedding, w)
    d2 = nb_sens_ave.reshape(-1,1)
    print('y component: ', np.mean(dd), np.mean(d2))
    labels = np.matmul(embedding, w) + w_s * nb_sens_ave.reshape(-1,1) # n x 1
    labels = labels.reshape(-1)
    labels_mean = np.mean(labels)
    labels_binary = np.zeros_like(labels)
    labels_binary[np.where(labels > labels_mean)] = 1.0

    print('pos labels: ', labels_binary.sum(), ' neg: ', len(labels_binary) - labels_binary.sum())

    # statistics
    pre_analysis(adj, labels, sens)

    data = {'x': features, 'adj': adj, 'labels': labels_binary, 'sens': sens,
            'z': embedding, 'v': v, 'feat_idxs': feat_idxs, 'alpha': alpha, 'w': w, 'w_s': w_s}
    scio.savemat(path, data)
    print('data saved in ', path)
    return data


def generate_synthetic_data(path, n, z_dim, p, q, alpha, beta, threshold, dim):
    sens = np.random.binomial(n=1, p=p, size=n)
    sens_repeat = np.repeat(sens.reshape(-1, 1), z_dim, axis=1)
    sens_embedding = np.random.normal(loc=sens_repeat, scale=1, size=(n, z_dim))
    labels = np.random.binomial(n=1, p=q, size=n)
    labels_repeat = np.repeat(labels.reshape(-1, 1), z_dim, axis=1)
    labels_embedding = np.random.normal(loc=labels_repeat, scale=1, size=(n, z_dim))
    features_embedding = np.concatenate((sens_embedding, labels_embedding), axis=1)
    weight = np.random.normal(loc=0, scale=1, size=(z_dim*2, dim))
    # features = np.matmul(features_embedding, weight)
    features = np.matmul(features_embedding, weight) + np.random.normal(loc=0, scale=1, size=(n, dim))

    adj = np.zeros((n, n))
    sens_sim = np.zeros((n, n))
    labels_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):  # i<=j
            if i == j:
                sens_sim[i][j] = -1
                labels_sim[i][j] = -1
                continue
            sens_sim[i][j] = sens_sim[j][i] = (sens[i] == sens[j])
            labels_sim[i][j] = labels_sim[j][i] = (labels[i] == labels[j])
            # sim_ij = 1 - spatial.distance.cosine(embedding[i], embedding[j])  # [-1, 1]
            # adj[i][j] = adj[j][i] = sim_ij + alpha * (sens[i] == sens[j])

    similarities = cosine_similarity(features_embedding)  # n x n
    similarities[np.arange(n), np.arange(n)] = -1
    adj = similarities + alpha * sens_sim + beta * labels_sim
    print('adj max: ', adj.max(), ' min: ', adj.min())
    adj[np.where(adj >= threshold)] = 1
    adj[np.where(adj < threshold)] = 0
    edge_num = adj.sum()
    adj = sparse.csr_matrix(adj)
    # features = np.concatenate((sens.reshape(-1,1), features), axis=1)

    # generate counterfactual
    sens_flip = 1 - sens
    sens_flip_repeat = np.repeat(sens_flip.reshape(-1, 1), z_dim, axis=1)
    sens_flip_embedding = np.random.normal(loc=sens_flip_repeat, scale=1, size=(n, z_dim))
    features_embedding = np.concatenate((sens_flip_embedding, labels_embedding), axis=1)
    features_cf = np.matmul(features_embedding, weight) + np.random.normal(loc=0, scale=1, size=(n, dim))

    adj_cf = np.zeros((n, n))
    sens_cf_sim = np.zeros((n, n))
    labels_cf_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                sens_cf_sim[i][j] = -1
                labels_cf_sim[i][j] = -1
                continue
            sens_cf_sim[i][j] = sens_cf_sim[j][i] = (sens_flip[i] == sens_flip[j])
            labels_cf_sim[i][j] = labels_cf_sim[j][i] = (labels[i] == labels[j])
    
    similarities_cf = cosine_similarity(features_cf)  # n x n
    similarities_cf[np.arange(n), np.arange(n)] = -1
    adj_cf = similarities_cf + alpha * sens_cf_sim + beta * labels_cf_sim
    print('adj_cf max', adj_cf.max(), ' min: ', adj_cf.min())
    adj_cf[np.where(adj_cf >= threshold)] = 1
    adj_cf[np.where(adj_cf < threshold)] = 0
    adj_cf = sparse.csr_matrix(adj_cf)
    # features_cf = np.concatenate((sens_flip.reshape(-1,1), features_cf), axis=1)

    # statistics
    # pre_analysis(adj, labels, sens)
    print('edge num: ', edge_num)
    data = {'x': features, 'adj': adj, 'labels': labels, 'sens': sens, 'x_cf': features_cf, 'adj_cf': adj_cf}
    # scio.savemat(path, data)
    print('data saved in ', path)
    return data

if __name__ == '__main__':
    n = 2000
    z_dim = 50
    p = 0.4
    q = 0.3
    alpha = 0.01  #
    beta = 0.01
    threshold = 0.6
    dim = 32

    dataset = 'synthetic'
    path_root = './dataset/synthetic/'
    path = path_root+f'synthetic_{n}_{z_dim}_{p}_{q}_{alpha}_{beta}_{threshold}_{dim}.mat'
    generate_synthetic_data(path, n, z_dim, p, q, alpha, beta, threshold, dim)




# edge count
# (pytorch) root@d3d3890e5fe5:~/projects/causal-fairness-public# python synthetic_generation.py 
# adj max:  0.7485439353207947  min:  -1.02
# adj_cf max 1.02  min:  -1.02
# edge num:  417930.0
# data saved in  ./dataset/synthetic/synthetic_2000_50_0.4_0.3_0.01_0.01_0.4.mat
# (pytorch) root@d3d3890e5fe5:~/projects/causal-fairness-public# python synthetic_generation.py 
# adj max:  0.7913202327923252  min:  -1.02
# adj_cf max 1.02  min:  -1.02
# edge num:  123642.0
# data saved in  ./dataset/synthetic/synthetic_2000_50_0.4_0.3_0.01_0.01_0.5.mat
# (pytorch) root@d3d3890e5fe5:~/projects/causal-fairness-public# python synthetic_generation.py 
# adj max:  0.7675836890785276  min:  -1.02
# adj_cf max 1.02  min:  -1.02
# edge num:  14408.0
# data saved in  ./dataset/synthetic/synthetic_2000_50_0.4_0.3_0.01_0.01_0.6.mat
# (pytorch) root@d3d3890e5fe5:~/projects/causal-fairness-public# python synthetic_generation.py 
# adj max:  0.7627960018969875  min:  -1.02
# adj_cf max 1.02  min:  -1.02
# edge num:  134.0