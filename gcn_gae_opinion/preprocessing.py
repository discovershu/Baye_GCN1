import numpy as np
import scipy.sparse as sp
import random
from collections import Counter
# from traffic_data.synthetic_opinion import *
import data.read_bigdata as rb_data

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features, placeholders, labels_b, label_un, labels_mask, omega, alpha_0, beta_0):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    feed_dict.update({placeholders['labels_b']: labels_b})
    feed_dict.update({placeholders['labels_un']: label_un})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['alpha_0']: alpha_0})
    feed_dict.update({placeholders['beta_0']: beta_0})
    feed_dict.update({placeholders['omega_t']: omega})
    return feed_dict


def construct_feed_dict_sub(adj_normalized, adj, features, placeholders, labels_1, label_2, labels_mask, label_3):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    feed_dict.update({placeholders['labels_1']: labels_1})
    feed_dict.update({placeholders['labels_2']: label_2})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['labels_3']: label_3})
    return feed_dict


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = range(edges.shape[0])
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def get_omega(b, u):
    W = 2.0
    a = 0.5
    d = 1.0 - b - u
    r = W * b / u
    s = W * d / u
    alpha = r + W * a
    beta = s + W * (1.0 - a)
    a0 = np.mean(alpha) * 1.0
    b0 = np.mean(beta) * 1.0
    # a0 = 2.0
    # b0 = 5.0
    omega = alpha / (alpha + beta)
    return omega, a0, b0


def get_omega_train(b, u, mask):
    W = 2.0
    a = 0.5
    d = 1.0 - b - u
    r = W * b / u
    s = W * d / u
    alpha = r + W * a
    beta = s + W * (1.0 - a)
    mask = np.array(mask, dtype=float)
    mask /= np.mean(mask)
    alpha *= mask
    beta *= mask
    a0 = np.mean(alpha) * 0.5
    b0 = np.mean(beta) * 0.5
    # a0 = 2
    # b0 = 9
    omega = alpha / (alpha + beta)
    return a0, b0

def mask_test_edge_opinion(test_rat, index):
    b_all = np.load("/network/rit/lab/ceashpc/xujiang/project/GAE_ICDM2018/gae/traffic_data/pa_belief_0.9.npy")
    u_all = np.load("/network/rit/lab/ceashpc/xujiang/project/GAE_ICDM2018/gae/traffic_data/pa_uncertain_0.9.npy")
    # belief, uncertain = rb_data.get_dc_data()
    belief = b_all[index]
    uncertain = u_all[index]
    belief = np.reshape(belief, [len(belief), 1])
    uncertain = np.reshape(uncertain, [len(uncertain), 1])
    omega, a0, b0 = get_omega(belief, uncertain)
    random.seed(132)
    test_num = int(test_rat * len(belief))
    test_index = random.sample(range(len(belief)), test_num)
    train_mask = np.zeros_like(belief, dtype=bool)
    test_mask = np.zeros_like(belief, dtype=bool)
    y_train_belief = np.zeros_like(belief)
    y_test_belief = np.zeros_like(belief)
    y_train_un = np.zeros_like(belief)
    y_test_un = np.zeros_like(belief)
    for i in range(len(test_mask)):
        if i in test_index:
            y_test_belief[i] = belief[i]
            y_test_un[i] = uncertain[i]
            test_mask[i] = True
        else:
            y_train_belief[i] = belief[i]
            y_train_un[i] = uncertain[i]
            train_mask[i] = True
    # a1, b1 = get_omega_train(belief, uncertain, train_mask)
    return y_train_belief, y_test_belief, y_train_un, y_test_un, train_mask, test_mask, omega, a0, b0


def mask_test_edge_epinion(test_rat, T):
    # b_all = np.load("/network/rit/lab/ceashpc/xujiang/project/GAE_TEST/gae/traffic_data/pa_belief_T38_0.8.npy")
    # u_all = np.load("/network/rit/lab/ceashpc/xujiang/project/GAE_TEST/gae/traffic_data/pa_uncertain_T38_0.8.npy")
    # # belief, uncertain = rb_data.get_dc_data()
    # belief = b_all[index]
    # uncertain = u_all[index]
    # belief, uncertain, test_index = rb_data.get_epinion_data(T)
    belief = np.load("./data/epinion_5000_0.8_T8_belief.npy")
    uncertain = np.load("./data/epinion_5000_0.8_T8_un.npy")
    test_index = np.load("./data/epinion_5000_0.8_T8_testindex.npy")
    belief = np.reshape(belief, [len(belief), 1])
    uncertain = np.reshape(uncertain, [len(uncertain), 1])
    omega, a0, b0 = get_omega(belief, uncertain)
    # random.seed(132)
    # test_num = int(test_rat * len(belief))
    # test_index = random.sample(range(len(belief)), test_num)
    train_mask = np.zeros_like(belief, dtype=bool)
    test_mask = np.zeros_like(belief, dtype=bool)
    y_train_belief = np.zeros_like(belief)
    y_test_belief = np.zeros_like(belief)
    y_train_un = np.zeros_like(belief)
    y_test_un = np.zeros_like(belief)
    for i in range(len(test_mask)):
        if i in test_index:
            y_test_belief[i] = belief[i]
            y_test_un[i] = uncertain[i]
            test_mask[i] = True
        else:
            y_train_belief[i] = belief[i]
            y_train_un[i] = uncertain[i]
            train_mask[i] = True
    return y_train_belief, y_test_belief, y_train_un, y_test_un, train_mask, test_mask, omega, a0, b0


def mask_test_syn_sub(test_rat, T):
    features_ = np.load("/network/rit/lab/ceashpc/xujiang/project/Dir_synthitic/label_38.npy")
    label = features_[:T]
    label = np.sum(label, axis=0) + 1
    random.seed(132)
    test_num = int(test_rat * len(label))
    test_index = random.sample(range(len(label)), test_num)
    label_1 = label[:, 0]
    label_2 = label[:, 1]
    label_3 = label[:, 2]

    label_1 = np.reshape(label_1, [len(label_1), 1])
    label_2 = np.reshape(label_2, [len(label_2), 1])
    label_3 = np.reshape(label_3, [len(label_3), 1])


    train_mask = np.zeros_like(label_1, dtype=bool)
    test_mask = np.zeros_like(label_1, dtype=bool)

    for i in range(len(test_mask)):
        if i in test_index:

            test_mask[i] = True
        else:

            train_mask[i] = True
    for i in range(len(test_mask)):
        if i in test_index:

            test_mask[i] = True
        else:

            train_mask[i] = True
    return label_1, label_2, label_3, train_mask, test_mask


def mask_test_syn_sub_dif(test_rat, T):
    label = np.load("/network/rit/lab/ceashpc/xujiang/project/Dir_synthitic/label_38_dif.npy")
    # label = features_[:T]
    # label = np.sum(label, axis=0) + 1

    random.seed(T)
    test_num = int(test_rat * len(label))
    test_index = random.sample(range(len(label)), test_num)
    label_1 = label[:, 0]
    label_2 = label[:, 1]
    label_3 = label[:, 2]

    label_1 = np.reshape(label_1, [len(label_1), 1])
    label_2 = np.reshape(label_2, [len(label_2), 1])
    label_3 = np.reshape(label_3, [len(label_3), 1])

    train_mask = np.zeros_like(label_1, dtype=bool)
    test_mask = np.zeros_like(label_1, dtype=bool)

    for i in range(len(test_mask)):
        if i in test_index:

            test_mask[i] = True
        else:

            train_mask[i] = True
    for i in range(len(test_mask)):
        if i in test_index:

            test_mask[i] = True
        else:

            train_mask[i] = True
    return label_1, label_2, label_3, train_mask, test_mask


def mask_test_syn_sub_dif2(test_rat, T):
    label = np.load("/network/rit/lab/ceashpc/xujiang/project/Dir_synthitic/label_38_dif.npy")
    # label = features_[:T]
    # label = np.sum(label, axis=0) + 1
    obs_num = np.sum(label, axis=1, dtype=float)
    obs_num = np.reshape(obs_num, [-1, 1])
    uncertainty = float(3) / obs_num
    obs_num = np.concatenate([obs_num, obs_num, obs_num], axis=1)

    label = label / obs_num
    random.seed(132)
    test_num = int(test_rat * len(label))
    test_index = random.sample(range(len(label)), test_num)
    label_1 = label[:, 0]
    label_2 = label[:, 1]
    label_3 = label[:, 2]

    label_1 = np.reshape(label_1, [len(label_1), 1])
    label_2 = np.reshape(label_2, [len(label_2), 1])
    label_3 = np.reshape(label_3, [len(label_3), 1])

    train_mask = np.zeros_like(label_1, dtype=bool)
    test_mask = np.zeros_like(label_1, dtype=bool)

    for i in range(len(test_mask)):
        if i in test_index:

            test_mask[i] = True
        else:

            train_mask[i] = True
    for i in range(len(test_mask)):
        if i in test_index:

            test_mask[i] = True
        else:

            train_mask[i] = True
    return label_1, label_2, label_3, train_mask, test_mask, uncertainty

def mask_test_edge_opinion_f(test_num, noise):
    # belief = np.load("/network/rit/lab/ceashpc/xujiang/project/GAE_TEST/gae/data/synthetic_belief_noise10.npy")
    # uncertain = np.load("/network/rit/lab/ceashpc/xujiang/project/GAE_TEST/gae/data/synthetic_uncertain_noise10.npy")
    _, belief = generate_synthetic_belief2(500, noise)
    _, uncertain = generate_synthetic_uncertain2(500, noise)
    features = np.zeros([len(belief), 2])
    features[:, 0] = belief
    features[:, 1] = uncertain
    belief = np.reshape(belief, [len(belief), 1])
    uncertain = np.reshape(uncertain, [len(uncertain), 1])
    omega = get_omega(belief, uncertain)
    random.seed(132)
    test_index = random.sample(range(len(belief)), test_num)
    train_mask = np.zeros_like(belief, dtype=bool)
    test_mask = np.zeros_like(belief, dtype=bool)
    y_train_belief = np.zeros_like(belief)
    y_test_belief = np.zeros_like(belief)
    y_train_un = np.zeros_like(belief)
    y_test_un = np.zeros_like(belief)
    for i in range(len(test_mask)):
        if i in test_index:
            features[i][0] = 0.0
            features[i][1] = 0.0
            y_test_belief[i] = belief[i]
            y_test_un[i] = uncertain[i]
            test_mask[i] = True
        else:
            y_train_belief[i] = belief[i]
            y_train_un[i] = uncertain[i]
            train_mask[i] = True
    features = sparse.csr_matrix(features)
    return y_train_belief, y_test_belief, y_train_un, y_test_un, train_mask, test_mask, omega, features

if __name__ == '__main__':
    mask_test_syn_sub_dif2(0.1, 1)

    print(1)