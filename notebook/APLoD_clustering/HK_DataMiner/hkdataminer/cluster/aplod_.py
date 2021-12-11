__author__ = 'LIU Song <liusong299@gmail.com>'
__contributors__ = "Lizhe ZHU, Tiago Lobato Gimenes, Xuhui HUANG"
__version__ = "0.91"
# Copyright (c) 2016, Hong Kong University of Science and Technology (HKUST)
# All rights reserved.
# ===============================================================================
# GLOBAL IMPORTS:
import random
import operator
import time
import numpy as np
from multiprocessing import Pool
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.utils.validation import check_is_fitted
from metrics.pairwise import pairwise_distances
from functools import reduce
# ===============================================================================
# LOCAL IMPORTS:
#import knn as knnn
# ===============================================================================

def FaissNearestNeighbors(X, eps, min_samples, nlist, nprobe, return_distance=False, IVFFlat=True, GPU=False):
    dimension = X.shape[1]
    if GPU is True:
        if IVFFlat is True:
            quantizer = faiss.IndexFlatL2(dimension)
            index_cpu = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
            # here we specify METRIC_L2, by default it performs inner-product search
            res = faiss.StandardGpuResources()  # use a single GPU
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
        n_samples = 10
        k = min_samples
        samples = np.random.choice(len(X), n_samples)
        # print(samples)
        D, I = index_gpu.search(X[samples], k)  # sanity check
        while np.max(D[:, k - 1]) < eps:
            k = k * 2
            D, I = index_gpu.search(X[samples], k)
            # print(np.max(D[:, k - 1]), k, eps)
        index_gpu.nprobe = nprobe
        D, I = index_gpu.search(X, k)  # actual search
    else:
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
        n_samples = 10
        k = min_samples
        samples = np.random.choice(len(X), n_samples)
        # print(samples)
        D, I = index_cpu.search(X[samples], k)  # sanity check
        while np.max(D[:, k - 1]) < eps:
            k = k * 2
            D, I = index_cpu.search(X[samples], k)
            # print(np.max(D[:, k - 1]), k, eps)
        index_cpu.nprobe = nprobe
        D, I = index_cpu.search(X, k)  # actual search
    if return_distance is True:
        return D, I
    else:
        return D

def run_knn(X, n_neighbors=100, n_samples=1000, metric='rmsd', algorithm='vp_tree'):
    #    X = check_array(X, accept_sparse='csr')
    #print "Calculating pairwise ", metric, " distances of ", n_samples, " samples..."
    t0 = time.time()
    if metric is "rmsd":
        samples = random.sample(list(X), n_samples)
        whole_samples= reduce(operator.add, (samples[i] for i in range(len(samples))))
    else:
        whole_samples = random.sample(list(X), n_samples)
    sample_dist_metric = pairwise_distances( whole_samples, whole_samples, metric=metric )
    t1 = time.time()
    #print "time:", t1-t0,
    #print "Done."

    # Calculate neighborhood for all samples. This leaves the original point
    # in, which needs to be considered later

    #print "Calculating knn..."
    t0 = time.time()
    if metric is 'rmsd':
        shape_x = np.shape(X.xyz)
        knn = knnn.vp_tree_parallel( np.reshape(X.xyz, (shape_x[0] * shape_x[1] * shape_x[2])), shape_x[1] * 3, "rmsd_serial" )
        distances_, indices = knn.query( np.linspace(0, len(X.xyz)-1, len(X.xyz), dtype='int'), n_neighbors )
    else:
        if algorithm is 'vp_tree':
            shape_x = np.shape(X)
            #print "shape_x:", shape_x
            knn = knnn.vp_tree_parallel( np.reshape(X, (shape_x[0] * shape_x[1])), shape_x[1], "euclidean_serial" )
            distances_, indices = knn.query( np.linspace(0, len(X)-1, len(X), dtype='int'), n_neighbors )
        else:
            neighbors_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
            neighbors_model.fit(X)
            distances_, indices = neighbors_model.kneighbors(X, n_neighbors=n_neighbors, return_distance=True)


    t1 = time.time()
    #print "time:", t1-t0,
    #print "Done."
    # Calculate distance between sample, and find dc
    # np.savetxt("./sample_dist_metric.txt", sample_dist_metric, fmt="%f")
    #np.savetxt("./distances_.txt", distances_, fmt="%f")
    #np.savetxt("./indices.txt", indices, fmt="%d")
    return sample_dist_metric, distances_, indices

def calculate_rho(X_len, n_neighbors, dc_2, indices, distances_):
    rho = np.zeros(X_len, dtype=np.float32)
    for i in range(0, X_len):
        for j in range(0,n_neighbors):
            index = indices[i,j]
            dist = distances_[i,j]
            gaussian = np.math.exp(-(dist**2/dc_2))
            rho[i] += gaussian
            rho[index] += gauss
    return rho

def aplod_clustering(X, weight=None, rho_cutoff=1.0, delta_cutoff=1.0, percent=0.1, n_neighbors=100, n_samples=1000,
                 metric='rmsd', algorithm='auto', sample_dist_metric=None, distances_=None, indices=None, parallel=True):
    """Adaptive Partitioning by Local Density-peaks (APLoD)
    We present an efficient density-based adaptive-resolution clustering method APLoD
    for analyzing large-scale molecular dynamics (MD) trajectories. APLoD performs the
    k-nearest-neighbors search to estimate the density of MD conformations in a local fashion,
    which can group MD conformations in the same high-density region into a cluster.
    APLoD greatly improves the popular density peaks algorithm by reducing the running
    time and the memory usage by 2-3 orders of magnitude for systems ranging from alanine
    dipeptide to a 370-residue Maltose-binding protein.  In addition, we demonstrate that
    APLoD can produce clusters with various sizes that are adaptive to the underlying density
    (i.e. larger clusters at low density regions, while smaller clusters at high density regions),
    which is a clear advantage over other popular clustering algorithms including k-centers and
    k-medoids. We anticipate that APLoD can be widely applied to split ultra-large MD datasets
    containing millions of conformations for subsequent construction of Markov State Models.

    Parameters
    ----------
    X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
        array of shape (n_samples, n_samples)
        A feature array, or array of distances between samples if
        ``metric='precomputed'``.

    rho_cutoff : float, optional
        The cut-off of the local density rho

    delta_cutoff : float, optional
        The cut-off of the distance delta from points of higher density
    percent : float, optional
        Average percentage of neighbours
    n_neighbors : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.calculate_distance for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.

    Returns
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers.
    labels_ :
        Labels of each point.

    Notes
    ----------
    NONE

    References
    ----------
    Song Liu, Lizhe Zhu and Xuhui Huang, "Adaptive Partitioning by Local
    ensity-peaks (APLoD): An efficient density-based clustering algorithm
    for analyzing molecular dynamics trajectories". Journal of Computational
    Chemistry XX/XX/2016: Vol.XXX No. XXX pp. XXXX-XXXX
    DOI: XXXX

    Alex Rodriguez, Alessandro Laio, "Clustering by fast search and find
    of APLoDs". Science 27 June 2014: Vol. 344 no. 6191 pp. 1492-1496
    DOI: 10.1126/science.1242072
    """

    if rho_cutoff < 0.0:
       raise ValueError("rho must be non negative!")
    if delta_cutoff < 0.0 and not delta_cutoff == None:
       raise ValueError("delta must be non negative!")

    if algorithm is not "precomputed":
            sample_dist_metric, distances_, indices = run_knn(X=X, n_neighbors=n_neighbors, n_samples=n_samples, metric=metric, algorithm=algorithm)
    else:
        if sample_dist_metric is None:
            raise ValueError("sample_dist_metric must not be None!")
        if distances_ is None:
            raise ValueError("distances must not be None!")
        if indices is None:
            raise ValueError("indices must not be None!")


    # Calculate distance between sample, and find dc
    sample_dist = []
    for i in range(0, n_samples):
        for j in range(i+1, n_samples):
            sample_dist.append(sample_dist_metric[i, j])
    sorted_sample_dist=np.sort(sample_dist)

    index = int(round(n_samples*percent))
    dc = sorted_sample_dist[index]

    # ----Cal rho: Gaussian kernel with neighbors
    # Calculating Rho using Gaussian kernel.
    X_len = len(X)
    rho = np.zeros(X_len, dtype=np.float32)
    dc_2 = dc**2

    if weight is None:
    #Don't have a weight
        for i in range(0, X_len):
            for j in range(0,n_neighbors):
                index = indices[i,j]
                dist = distances_[i,j]
                gaussian = np.math.exp(-(dist**2/dc_2))
                rho[i] += gaussian
                rho[index] += gaussian
    else:
        for i in range(0, X_len):
            for j in range(0,n_neighbors):
                index = indices[i,j]
                dist = distances_[i,j]
                gaussian = np.math.exp(-(dist**2/dc_2)) * weight[j]
                rho[i] += gaussian
                rho[index] += gaussian

    # Calculating Highest Rho and Max Distance.
    rho_argsorted = np.argsort(rho)[::-1]
    max_rho = rho[rho_argsorted[0]]
    max_dist = np.max(distances_)

    if delta_cutoff == None:
        delta_cutoff = max_dist
    delta = [max_dist for i in range(X_len)]
    nneigh = [i for i in range(X_len)]

    # Calculating Delta.
    for i in range(0, X_len):
        for j in range(0, n_neighbors):
            index = indices[i, j]
            if rho[index] > rho[i]:
                delta[i] = distances_[i, j]
                nneigh[i] = index
                break

    # Clustering Data.
    # Initially, all samples are noise.
    
    rho_cut_index = int(1.0 * X_len) - 1
    rho_cut = rho[rho_argsorted[rho_cut_index]]
    distances_sorted = np.sort(distances_, axis=None)
    
    distances_index = int(1.0 * len(distances_sorted)) - 1
    delta_cut = distances_sorted[distances_index]
    #print "rho_cut:", rho_cut, rho_cut_index
    #print "delta_cut", delta_cut, distances_index
 
    if metric is "rmsd:":
        labels_ = -np.ones(X.xyz.shape[0], dtype=np.intp)
    else:
        labels_ = -np.ones(X_len, dtype=np.intp)
    cluster_centers_list = []
    cluster = 0
    for i in range(len(rho_argsorted)):
        index = rho_argsorted[i]
        if labels_[index] == -1 and delta[index] >= delta_cut and rho[index] > rho_cut:
            labels_[index] = cluster
            cluster_centers_list.append(index)
            cluster += 1
        else:
            if labels_[index] == -1 and labels_[nneigh[index]] != -1:
                labels_[index] = labels_[nneigh[index]]

    cluster_centers_ = cluster_centers_list
    return cluster_centers_, labels_



class APLoD(BaseEstimator, ClusterMixin):
    """Adaptive Partitioning by Local Density-peaks (APLoD)
    We present an efficient density-based adaptive-resolution clustering method APLoD
    for analyzing large-scale molecular dynamics (MD) trajectories. APLoD performs the
    k-nearest-neighbors search to estimate the density of MD conformations in a local fashion,
    which can group MD conformations in the same high-density region into a cluster.
    APLoD greatly improves the popular density peaks algorithm by reducing the running
    time and the memory usage by 2-3 orders of magnitude for systems ranging from alanine
    dipeptide to a 370-residue Maltose-binding protein.  In addition, we demonstrate that
    APLoD can produce clusters with various sizes that are adaptive to the underlying density
    (i.e. larger clusters at low density regions, while smaller clusters at high density regions),
    which is a clear advantage over other popular clustering algorithms including k-centers and
    k-medoids. We anticipate that APLoD can be widely applied to split ultra-large MD datasets
    containing millions of conformations for subsequent construction of Markov State Models.

    Parameters
    ----------
    rho_cutoff : float, optional
       The cut-off of the local density rho

    delta_cutoff : float, optional
        The cut-off of the distance delta from points of higher density
    percent : float, optional
        Average percentage of neighbours
    n_neighbors : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.calculate_distance for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.

    Returns
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers.
    labels_ :
        Labels of each point.

    Notes
    ----------
    NONE

    References
    ----------
    Song Liu, Lizhe Zhu and Xuhui Huang, "Adaptive Partitioning by Local
    ensity-peaks (APLoD): An efficient density-based clustering algorithm
    for analyzing molecular dynamics trajectories". Journal of Computational
    Chemistry XX/XX/2016: Vol.XXX No. XXX pp. XXXX-XXXX
    DOI: XXXX

    Alex Rodriguez, Alessandro Laio, "Clustering by fast search and find
    of APLoDs". Science 27 June 2014: Vol. 344 no. 6191 pp. 1492-1496
    DOI: 10.1126/science.1242072
    """
    def __init__(self, rho_cutoff=1.0, delta_cutoff=1.0, percent=0.2, n_neighbors=100, n_samples=1000,
                 metric='rmsd', algorithm='auto', sample_dist_metric=None, distances_=None, indices=None, parallel=True):
        self.rho_cutoff = rho_cutoff
        self.delta_cutoff = delta_cutoff
        self.percent = percent
        self.n_neighbors = n_neighbors
        self.n_samples = n_samples
        self.metric = metric
        self.algorithm = algorithm
        self.sample_dist_metric = sample_dist_metric
        self.distances_ = distances_
        self.indices = indices
        self.parallel = parallel

    def fit(self, X, weight=None):
        """Perform clustering.
        Parameters
        -----------
        X : array-like, shape=[n_samples, n_features]
            Samples to cluster.
        """
#        X = check_array(X)
        print( "Doing APLoD clustering..." )
        t0 = time.time()
        self.cluster_centers_, self.labels_ = \
            aplod_clustering(X, weight=weight, rho_cutoff=self.rho_cutoff, delta_cutoff=self.delta_cutoff,
                         percent=self.percent, n_neighbors=self.n_neighbors,n_samples=self.n_samples,
                         metric=self.metric, algorithm=self.algorithm,sample_dist_metric=self.sample_dist_metric,
                         distances_=self.distances_, indices=self.indices, parallel=self.parallel)
        t1 = time.time()
        print( "APLoD Time Cost:", t1-t0 )
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[n_samples, n_features]
            New data to predict.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, "cluster_centers_")

        return pairwise_distances_argmin(X, self.cluster_centers_)

