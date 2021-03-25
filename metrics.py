import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import entropy
from sklearn.metrics import mutual_info_score


def nmi_score(y_true, y_pred):
    """NMI
    https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    this function is not the same with
    sklearn.metrics.normalized_mutual_info_score
    in that this function uses [H(y_true)+H(y_pred)/2] while
    that of sklearn uses sqrt(H(y_true)H(y_pred))
    """
    labels = align_labels(y_true)
    mi = mutual_info_score(y_true, y_pred)
    h1 = entropy(labels)
    h2 = entropy(y_pred)
    return 2 * mi / (h1 + h2)


def align_labels(labels, return_reference=False):
    """align sparse labels
    Args:
        labels
        return_reference
    Returns:
        labels
        or
        labels, refenrence
    example:
    >> a = [1, 2, 1, 1] # lack of 0
    >> a = align_labels(a)
    >> a # [0, 1, 0, 0]
    """
    reference = list()
    labels_dense = np.ones_like(labels) * -1
    for i, label in enumerate(np.unique(labels)):
        labels_dense[labels == label] = i
        reference.append(label)

    assert np.all(labels_dense >= 0)

    if return_reference:
        return labels_dense, reference
    else:
        return labels_dense


def cluster_accuracy(y_true, y_pred, return_details=False):
    """accuracy"""
    # y_true sometimes sparse
    y_true_dense = align_labels(y_true)
    # conf_mat[i, j]: label is i, while pred is j
    conf_mat = confusion_matrix(y_true_dense, y_pred)
    # assign label to pred (cluster)
    cost_mat = -conf_mat.T
    # workers=cluster, jobs=label
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    acc = -cost_mat[row_ind, col_ind].sum() / y_true_dense.size

    if return_details:
        assert np.all(row_ind == np.arange(np.unique(y_true_dense).size))
        details = {
            "CM": conf_mat,
            "label_assign": col_ind,
            "sorted_CM": conf_mat[col_ind],
        }
        return acc, details
    else:
        return acc
