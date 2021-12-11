# -*- coding: utf-8 -*-
"""
DBSCAN Acclerated by Facebook AI Faiss
DBSCAN: Density-Based Spatial Clustering of Applications with Noise
"""

# Author: Robert Layton <robertlayton@gmail.com>
#         Joel Nothman <joel.nothman@gmail.com>
#         Lars Buitinck
#
# License: BSD 3 clause

import numpy as np
import time
from scipy import sparse
from numba import autojit
import numba

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array, check_consistent_length
#from sklearn.neighbors import NearestNeighbors

from sklearn.cluster._dbscan_inner import dbscan_inner
import faiss

@autojit
def get_neighborhoods(D, I, eps):
    neighborhoods = []
    for i in range(len(D)):
        distances = D[i]
        #print(distances)
        distances = np.delete(distances, 0)
        indices = I[i]
        indices = np.delete(indices, 0)
        #print(indices)
        index = indices[distances <= eps]
        neighborhoods.append(index)
    #neighborhoods = np.asarray(neighborhoods)
    #np.savetxt('faiss_neighborhoods', np.asarray(neighborhoods), fmt='%s')
    return np.asarray(neighborhoods)

def cpu_radius_neighbors(X, eps, min_samples, nlist, nprobe, return_distance=False, IVFFlat=True):
    dimension = X.shape[1]
    if IVFFlat is True:
        quantizer = faiss.IndexFlatL2(dimension)
        index_cpu = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        # here we specify METRIC_L2, by default it performs inner-product search
        assert not index_cpu.is_trained
        index_cpu.train(X)
        assert index_cpu.is_trained
        # here we specify METRIC_L2, by default it performs inner-product search
    else:
        index_cpu = faiss.IndexFlatL2(dimension)
    index_cpu.add(X)
    n_samples = 1000
    k = min_samples
    samples = np.random.choice(len(X), n_samples)
    # print(samples)
    D, I = index_cpu.search(X[samples], k)  # sanity check
    while np.min(np.amax(D, axis=1)) < eps:
        k = k * 2
       # D, I = index_gpu.search(X[samples], k)
        #print(np.min(np.amax(D, axis=1)), eps, k)
        D, I = index_cpu.search(X[samples], k)
    if k > 1024:
        k = 1000
        #print(np.max(D[:, k - 1]), k, eps)
    index_cpu.nprobe = nprobe
    D, I = index_cpu.search(X, k)  # actual search

    return get_neighborhoods(D, I, eps)


def gpu_radius_neighbors(X, eps, min_samples, nlist, nprobe, return_distance=False, IVFFlat=True):
    dimension = X.shape[1]
    if IVFFlat is True:
        quantizer = faiss.IndexFlatL2(dimension)
        index_cpu = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        # here we specify METRIC_L2, by default it performs inner-product search
        res = faiss.StandardGpuResources() # use a single GPU
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        # make it an IVF GPU index
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        assert not index_gpu.is_trained
        index_gpu.train(X)
        assert index_gpu.is_trained
        # here we specify METRIC_L2, by default it performs inner-product search
    else:
        index_cpu = faiss.IndexFlatL2(dimension)
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    index_gpu.add(X)
    n_samples = 1000
    k = min_samples
    samples = np.random.choice(len(X), n_samples)
    # print(samples)
    D, I = index_gpu.search(X[samples], k)  # sanity check
    while np.max(D[:, k - 1]) < eps:
        k = k * 2
        D, I = index_gpu.search(X[samples], k)
        #print(np.max(D[:, k - 1]), k, eps)
    index_gpu.nprobe = nprobe
    D, I = index_gpu.search(X, k)  # actual search

    return get_neighborhoods(D, I, eps)

def faiss_dbscan(X, eps=0.5, min_samples=5, nlist=100, nprobe=5, metric='l2', metric_params=None,
           algorithm='auto', leaf_size=30, p=2, sample_weight=None, n_jobs=1, GPU=False, IVFFlat=True):
    """Perform DBSCAN clustering from vector array or distance matrix.

    Read more in the :ref:`User Guide <dbscan>`.

    Parameters
    ----------
    X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
            array of shape (n_samples, n_samples)
        A feature array, or array of distances between samples if
        ``metric='precomputed'``.

    eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.

    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.pairwise_distances for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a sparse matrix, in which case only "nonzero"
        elements may be considered neighbors for DBSCAN.

    metric_params : dict, optional
        Additional keyword arguments for the metric function.

        .. versionadded:: 0.19

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.

    p : float, optional
        The power of the Minkowski metric to be used to calculate distance
        between points.

    sample_weight : array, shape (n_samples,), optional
        Weight of each sample, such that a sample with a weight of at least
        ``min_samples`` is by itself a core sample; a sample with negative
        weight may inhibit its eps-neighbor from being core.
        Note that weights are absolute, and default to 1.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    Returns
    -------
    core_samples : array [n_core_samples]
        Indices of core samples.

    labels : array [n_samples]
        Cluster labels for each point.  Noisy samples are given the label -1.

    Notes
    -----
    See examples/cluster/plot_dbscan.py for an example.

    This implementation bulk-computes all neighborhood queries, which increases
    the memory complexity to O(n.d) where d is the average number of neighbors,
    while original DBSCAN had memory complexity O(n).

    Sparse neighborhoods can be precomputed using
    :func:`NearestNeighbors.radius_neighbors_graph
    <sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>`
    with ``mode='distance'``.

    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander, and X. Xu, "A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise".
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996
    """
    if not eps > 0.0:
        raise ValueError("eps must be positive.")

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        check_consistent_length(X, sample_weight)

    # Calculate neighborhood for all samples. This leaves the original point
    # in, which needs to be considered later (i.e. point i is in the
    # neighborhood of point i. While True, its useless information)
    if GPU is True:
        neighborhoods = gpu_radius_neighbors(X, eps, min_samples, nlist, nprobe, return_distance=False, IVFFlat=IVFFlat)
    else:
        neighborhoods = cpu_radius_neighbors(X, eps, min_samples, nlist, nprobe, return_distance=False, IVFFlat=IVFFlat)
    if sample_weight is None:
        n_neighbors = np.array([len(neighbors)
                                for neighbors in neighborhoods])
    else:
        n_neighbors = np.array([np.sum(sample_weight[neighbors])
                                for neighbors in neighborhoods])

    # Initially, all samples are noise.
    labels = -np.ones(X.shape[0], dtype=np.intp)

    # A list of all core samples found.
    core_samples = np.asarray(n_neighbors >= min_samples, dtype=np.uint8)
    dbscan_inner(core_samples, neighborhoods, labels)
    return np.where(core_samples)[0], labels

class Faiss_DBSCAN(BaseEstimator, ClusterMixin):
    """Perform DBSCAN clustering from vector array or distance matrix.

    DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
    Finds core samples of high density and expands clusters from them.
    Good for data which contains clusters of similar density.

    Read more in the :ref:`User Guide <dbscan>`.

    Parameters
    ----------
    eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.

    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.calculate_distance for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a sparse matrix, in which case only "nonzero"
        elements may be considered neighbors for DBSCAN.

        .. versionadded:: 0.17
           metric *precomputed* to accept precomputed sparse matrix.

    metric_params : dict, optional
        Additional keyword arguments for the metric function.

        .. versionadded:: 0.19

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.

    p : float, optional
        The power of the Minkowski metric to be used to calculate distance
        between points.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    Attributes
    ----------
    core_sample_indices_ : array, shape = [n_core_samples]
        Indices of core samples.

    components_ : array, shape = [n_core_samples, n_features]
        Copy of each core sample found by training.

    labels_ : array, shape = [n_samples]
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.

    Notes
    -----
    See examples/cluster/plot_dbscan.py for an example.

    This implementation bulk-computes all neighborhood queries, which increases
    the memory complexity to O(n.d) where d is the average number of neighbors,
    while original DBSCAN had memory complexity O(n).

    Sparse neighborhoods can be precomputed using
    :func:`NearestNeighbors.radius_neighbors_graph
    <sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>`
    with ``mode='distance'``.

    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander, and X. Xu, "A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise".
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996
    """

    def __init__(self, eps=0.5, min_samples=5, nlist=100, nprobe=5, metric='l2', n_jobs=1, GPU=False, IVFFlat=True):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.n_jobs = n_jobs
        self.GPU = GPU
        self.IVFFlat = IVFFlat
        self.nlist = nlist
        self.nprobe = nprobe

    def fit(self, X, y=None, sample_weight=None):
        """Perform DBSCAN clustering from features or distance matrix.

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.
        sample_weight : array, shape (n_samples,), optional
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with negative
            weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.
        """
        #if metric is not "rmsd":
        #    X = check_array(X, accept_sparse='csr')
        #t0 = time.time()
        clust = faiss_dbscan(X, eps=self.eps, min_samples=self.min_samples, nlist=self.nlist, nprobe=self.nprobe, sample_weight=sample_weight, GPU=self.GPU, IVFFlat=self.IVFFlat)
        #t1 = time.time()
        #print("Faiss DBSCAN clustering Time Cost:", t1 - t0)
        self.core_sample_indices_, self.labels_ = clust
        if len(self.core_sample_indices_):
            # fix for scipy sparse indexing issue
            self.components_ = X[self.core_sample_indices_].copy()
        else:
            # no core samples
            self.components_ = np.empty((0, X.shape[1]))
        return self
