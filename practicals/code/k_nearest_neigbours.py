from scipy.spatial import distance_matrix
from scipy import stats
import numpy as np


def comp_distance_matrix(A, B, p=2):
    """
    Compute all pairwise distances between vectors in A and B.
    :param A: Input data matrix
    :param B: Input data matrix
    :param p: p-norm to use
    :return: Distance matrix
    """
    distances = distance_matrix(A, B, p)
    return distances


def find_k_nearest_neighbors(distances, k):
    """
    :param k: How many neighbor points K we wish to include
    :param distances: distance matrix (n_train, n_test) between training and test points
    :return: indices of the k nearest neighbors
    """
    # get the indices of the k nearest neighbors
    k_nearest_neighbors = np.argsort(distances, axis=1)[:, :k]
    return k_nearest_neighbors


def predict_topk(k_nearest_neighbors, labels):
    """
    :param k_nearest_neighbors: indices of the k nearest neighbors
    :param labels: labels of the training data
    :return: predicted labels
    """
    # get the labels of the k nearest neighbors
    k_nearest_labels = labels[k_nearest_neighbors]
    # get the most common label
    predicted_labels = stats.mode(k_nearest_labels, axis=1)[0]
    return predicted_labels