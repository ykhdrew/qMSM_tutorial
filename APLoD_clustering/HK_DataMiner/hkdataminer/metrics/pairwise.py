__author__ = 'stephen'
import numpy as np
import mdtraj as md
import sklearn.metrics.pairwise as sp
def pairwise_distances(X, Y=None, index=None, metric="euclidean"):
    '''
    Compute the distance matrix from a vector array X and optional Y.
    This method takes either a vector array or a distance matrix,
    and returns a distance matrix. If the input is a vector array,
    the distances are computed. If the input is a distances matrix,
    it is returned instead.
    This method provides a safe way to take a distance matrix as input,
    while preserving compatibility with many other algorithms that take
    a vector array.

    :param X:  array [n_samples_a, n_samples_a]
        Array of pairwise distances between samples, or a feature array.
    :param Y:   array [n_samples_b, n_features]
        A second feature array only if X has shape [n_samples_a, n_features].
    :param index:  int, the index of element in X array
    :param metric: The metric to use when calculating distance between instances in a feature array.
        If metric ='rmsd', it should be computed by MDTraj
    :return: The distances
    '''
    if metric == "rmsd":
        if Y is None:
            distances_ = md.rmsd(X, X, index, parallel=True, precentered=True)
        else:
            #distances_ = np.empty((len(X), len(Y)), dtype=np.float32)
           # for i in xrange(len(Y)):
            distances_ = md.rmsd(X, Y, index, parallel=True, precentered=True)
        return distances_
    else:
        if Y is None:
            #print "if Y is None"
            return sp.pairwise_distances(X, X[index], metric=metric, n_jobs=1)
        if index is None:
            #print "if index is None, pairwise XX"
            return sp.pairwise_distances(X, X, metric=metric, n_jobs=1)
