import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from karateclub import EgoNetSplitter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import sys


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def save_file(filepath, content):
    with open(filepath, 'w') as f:
        f.write(str(content))
    f.close()


def load_file(filepath):
    with open(filepath, 'r') as f:
        content = eval(f.read())
    f.close()
    return content


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    # load dataset
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
        for i in range(len(ty)):
            if max(ty[i]) == 0:
                ty[i][0] = 1

    # resort nodes
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # format features and labels of nodes
    mid_features = features.toarray().tolist()
    mid_labels = np.where(labels)[1].tolist()

    # cluster voter nodes
    g = nx.from_dict_of_lists(graph)
    g_s_l = nx.connected_components(g)
    cluster = {}
    for g_s in g_s_l:
        if len(g_s) > 10:
            id_map_list = []
            id_sort = sorted(list(g_s))
            for i in range(len(id_sort)):
                id_map_list.append(id_sort[i])
            sub_g_nid = {}
            for node in graph:
                if node in id_map_list:
                    n_l = []
                    for n_id in graph[node]:
                        n_l.append(id_map_list.index(n_id))
                    sub_g_nid[id_map_list.index(node)] = n_l
            sub_g = nx.Graph(sub_g_nid)
            splitter = EgoNetSplitter(1.0)
            splitter.fit(sub_g)
            res = splitter.get_memberships()
            sub_c = {}
            for n_id in res:
                o_n_id = id_map_list[n_id]
                for c_id in res[n_id]:
                    if c_id not in sub_c:
                        sub_c[c_id] = []
                    sub_c[c_id].append(o_n_id)
            for c_id in sub_c:
                cluster[len(cluster)] = sub_c[c_id]
        else:
            cluster[len(cluster)] = list(g_s)

    # add losing edges
    for n_id_1 in graph:
        for n_id_2 in graph[n_id_1]:
            p = 0
            for c_id in cluster:
                if n_id_1 in cluster[c_id] and n_id_2 in cluster[c_id]:
                    p = 1
                    break
            if p == 0:
                cluster[len(cluster)] = [n_id_1, n_id_2]

    # merge clusters on each node
    cluster_n = {}
    for i in range(len(mid_labels)):
        n_s = []
        for c_id in cluster:
            if i in cluster[c_id]:
                if 2 < len(cluster[c_id]):
                    cluster_n[len(cluster_n)] = cluster[c_id]
                else:
                    n_s += cluster[c_id]
        n_s = set(n_s)
        if 2 < len(n_s):
            cluster_n[len(cluster_n)] = list(n_s)
    cluster = cluster_n

    if dataset_str == 'pubmed':
        cluster_n = {}
        for c_id in cluster:
            if len(cluster[c_id]) < 10:
                cluster_n[len(cluster_n)] = cluster[c_id]
        cluster = cluster_n

    # insert edges into edge set
    for i in range(len(cluster)):
        graph[i + len(ally) + len(ty)] = list(cluster[i])

    print("Cluster Completed.")

    # split elector nodes to train/test sets
    c_train_x = []
    c_train_y = []
    c_train_id_l = []
    c_test_x = []
    c_test_y = []
    c_test_id_l = []
    c_features = []

    for c_id in cluster: # obtain attributes and labels of elector nodes
        c_f = [0.0] * len(mid_features[0])
        c_l = [0] * (max(mid_labels) + 1)
        c_train_l = [0] * (max(mid_labels) + 1)
        p_train = 0
        for n_id in cluster[c_id]: # statistic nodes in each cluster
            c_f = list(map(lambda x: x[0] + x[1], zip(c_f, mid_features[n_id])))
            c_l[mid_labels[n_id]] += 1
            if n_id < len(y): # inherit labels from voter nodes in training set
                c_train_l[mid_labels[n_id]] += 1
                p_train = 1
        c_features.append(c_f) # avg of attributes of voter nodes
        max_l = max(c_l)
        max_train_l = max(c_train_l)
        if dataset_str != 'pubmed':
            if p_train == 1 and c_train_l.count(max_train_l) == 1 and max_train_l != 1: # elector node has corresponding members in training set and can inherit labels (only one max label and its times > 1)
                c_train_x.append(c_f)
                c_train_y.append(c_train_l.index(max_train_l))
                c_train_id_l.append(c_id)
            else:
                c_test_x.append(c_f)
                c_test_y.append(c_l.index(max_l))
                c_test_id_l.append(c_id)
        else:
            if p_train == 1 and c_train_l.count(max_train_l) == 1:
                c_train_x.append(c_f)
                c_train_y.append(c_train_l.index(max_train_l))
                c_train_id_l.append(c_id)
            else:
                c_test_x.append(c_f)
                c_test_y.append(c_l.index(max_l))
                c_test_id_l.append(c_id)

    # for i in range(len(mid_features)):
    #     if i < len(y):
    #         c_train_x.append(mid_features[i])
    #         c_train_y.append(mid_labels[i])
    #         c_train_id_l.append(i)
    #     else:
    #         c_test_x.append(mid_features[i])
    #         c_test_y.append(mid_labels[i])
    #         c_test_id_l.append(i)

    # classify the elector nodes
    skip = 0
    if skip == 0:
        pred_c_test_y = []
        pred_c_test_y_prob = []
        for i in range(10):
            if len(pred_c_test_y) != 0:
                for c_id in c_test_id_l:
                    if max(pred_c_test_y_prob[c_test_id_l.index(c_id)]) > 0.99 and c_id not in c_train_id_l:
                        c_train_x.append(c_test_x[c_test_id_l.index(c_id)])
                        c_train_y.append(c_test_y[c_test_id_l.index(c_id)])
                        c_train_id_l.append(c_id)

            model_init = GradientBoostingClassifier()
            model_init.set_params(learning_rate=0.25, max_depth=5)
            model_init.fit(c_train_x, c_train_y)
            pred_c_test_y = model_init.predict(c_test_x)
            pred_c_test_y_prob = model_init.predict_proba(c_test_x)
            print("{}/10 th iteration...".format(str(i+1)))

        cont = {'c_train_x': c_train_x, 'c_train_y': c_train_y, 'c_train_id_l': c_train_id_l, 'c_test_x': c_test_x,
                'c_test_y': c_test_y, 'c_test_id_l': c_test_id_l}
        save_file("data/{}.content.save".format(dataset_str), cont)
    else:
        cont = load_file("data/{}.content.save".format(dataset_str))
        c_train_x = cont['c_train_x']
        c_train_y = cont['c_train_y']
        c_train_id_l = cont['c_train_id_l']
        c_test_x = cont['c_test_x']
        c_test_y = cont['c_test_y']
        c_test_id_l = cont['c_test_id_l']

    # obtain final elector node labels (obs, pred, remain)
    model = GradientBoostingClassifier()
    model.set_params(learning_rate=0.25, max_depth=5)
    model.fit(c_train_x, c_train_y)
    pred_c_test_y = model.predict(c_test_x)
    pred_c_test_y_prob = model.predict_proba(c_test_x)
    print("Final elector node classification accuracy: ", accuracy_score(c_test_y, pred_c_test_y))

    # splice voter nodes and elector nodes
    # insert new train samples
    n_features = features.toarray().tolist()
    n_labels = labels.tolist()
    extra_train = []

    for i in range(len(c_features)):
        n_features.append(c_features[i])
        n_labels.append([0] * len(labels[0]))
    for i in range(len(c_features)):
        x = i + len(ty) + len(ally)
        if i in c_train_id_l:
            n_labels[x][c_train_y[c_train_id_l.index(i)]] = 1 # complete labels of elector nodes with (obs, pred(train))
            extra_train.append(x) # insert to training set
        if i in c_test_id_l:
            n_labels[x][pred_c_test_y[c_test_id_l.index(i)]] = 1 # complete labels of elector nodes with (pred(test > 0.99), remain(test < 0.99))
            if max(pred_c_test_y_prob[c_test_id_l.index(i)]) > 0.99 and x not in extra_train: # insert to training set
                extra_train.append(x)

    out_file = 0
    if out_file == 1:
        with open('{}.extra'.format(dataset_str), 'w') as f:
            f.write(str(extra_train))
        f.close()
        with open('{}.cites'.format(dataset_str), 'w') as f:
            out_edges = ""
            for n_id_f in graph:
                for n_id_b in graph[n_id_f]:
                    out_edges += "{}\t{}\n".format(n_id_f, n_id_b)
            f.write(out_edges)
        f.close()
        with open('{}.content'.format(dataset_str), 'w') as f:
            out_content = ""
            for i in range(len(n_features)):
                out_content += "{}\t".format(str(i))
                for feat in n_features[i]:
                    out_content += "{}\t".format(str(feat))
                out_content += "{}\n".format(str(n_labels[i].index(1)))
            f.write(out_content)
        f.close()

    # reformat data
    features = sp.csr_matrix(np.array(n_features))
    features = normalize_features(features)
    features = features.tolil()

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.array(n_labels)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y))) + extra_train
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
