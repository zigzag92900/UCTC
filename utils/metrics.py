from scipy.optimize import linear_sum_assignment
import numpy as npy


def cluster_acc(y_true, y_pred):
    assert y_pred.size == y_true.size
    D = int(max(y_pred.max(), y_true.max())) + 1
    w = npy.zeros((D, D), dtype=npy.int64)
    for i in range(y_pred.size):
        w[int(y_pred[i]), int(y_true[i])] += 1
    inx = npy.transpose(npy.array(linear_sum_assignment(w.max() - w)))
    return sum(w[i, j] for i, j in inx) * 1. / y_pred.size


def cluster_nmi(y_true, y_pred):
    from sklearn.metrics.cluster.supervised import normalized_mutual_info_score
    return normalized_mutual_info_score(y_true, y_pred)


def cluster_ari(y_true, y_pred):
    from sklearn.metrics.cluster.supervised import adjusted_rand_score
    return adjusted_rand_score(y_true, y_pred)


def cluster_ri(y_true, y_pred):
    assert y_pred.size == y_true.size
    TP = 0
    TN = 0
    NUM = 0
    for i in range(y_pred.size):
        for j in range(i):
            TP += int(y_pred[i] == y_pred[j] and y_true[i] == y_true[j])
            TN += int(y_pred[i] != y_pred[j] and y_true[i] != y_true[j])
            NUM += 1
    return (TP+TN)/NUM


def cluster_ri_(y_true, y_pred):
    assert y_pred.size == y_true.size
    y_true = y_true.astype(npy.int32)
    y_pred = y_pred.astype(npy.int32)
    if y_true.min() > 0:
        y_true -= y_true.min()
    k = int(y_true.max()) + 1
    n = y_true.shape[0]
    x = npy.eye(k)[y_true]
    y = npy.eye(k)[y_pred]
    x = x @ x.T
    y = y @ y.T
    return (npy.sum(x == y)-n)/(n*n-n)
