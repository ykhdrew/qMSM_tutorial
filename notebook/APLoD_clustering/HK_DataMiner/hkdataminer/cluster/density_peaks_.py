__author__ = 'stephen'
#===============================================================================
# GLOBAL IMPORTS:
import os, sys
import numpy as np
import time
import random
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import pairwise_distances
# ==============================================
#===============================================================================
# LOCAL IMPORTS:
HK_DataMiner_Path = os.path.relpath(os.pardir)
sys.path.append(HK_DataMiner_Path)
#===============================================================================

def density_peaks_clustering(X, n_clusters=3, rho_cutoff=1.0, delta_cutoff=1.0, percent=0.1,
                     metric='eculidean', parallel=True):
    """Adaptive Partitioning by Local Density-peaks (density_peaks)
    We present an efficient density-based adaptive-resolution clustering method density_peaks
    for analyzing large-scale molecular dynamics (MD) trajectories. density_peaks performs the
    k-nearest-neighbors search to estimate the density of MD conformations in a local fashion,
    which can group MD conformations in the same high-density region into a cluster.
    density_peaks greatly improves the popular density peaks algorithm by reducing the running
    time and the memory usage by 2-3 orders of magnitude for systems ranging from alanine
    dipeptide to a 370-residue Maltose-binding protein.  In addition, we demonstrate that
    density_peaks can produce clusters with various sizes that are adaptive to the underlying density
    (i.e. larger clusters at low density regions, while smaller clusters at high density regions),
    which is a clear advantage over other popular clustering algorithms including k-centers and
    k-medoids. We anticipate that density_peaks can be widely applied to split ultra-large MD datasets
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
    ensity-peaks (density_peaks): An efficient density-based clustering algorithm
    for analyzing molecular dynamics trajectories". Journal of Computational
    Chemistry XX/XX/2016: Vol.XXX No. XXX pp. XXXX-XXXX
    DOI: XXXX

    Alex Rodriguez, Alessandro Laio, "Clustering by fast search and find
    of density_peakss". Science 27 June 2014: Vol. 344 no. 6191 pp. 1492-1496
    DOI: 10.1126/science.1242072
    """

    if rho_cutoff < 0.0:
        raise ValueError("rho must be non negative!")
    if delta_cutoff < 0.0 and not delta_cutoff == None:
        raise ValueError("delta must be non negative!")

    X_len = len(X)
    # Calculate distance and find dc
    distances_ = pairwise_distances(X, X, metric=metric, n_jobs=1)

    sorted_dist = np.sort(distances_)

    index = int(round(X_len * percent))
    dc = sorted_dist[index]

    # ----Cal rho: Gaussian kernel with neighbors
    # Calculating Rho using Gaussian kernel.

    rho = np.zeros(X_len, dtype=np.float32)
    dc_2 = dc ** 2

    for i in xrange(0, X_len-1):
        for j in xrange(i+1, X_len):
            dist = distances_[i, j]
            gaussian = np.math.exp(-(dist ** 2 / dc_2))
            rho[i] += gaussian
            rho[j] += gaussian

    # Calculating Highest Rho and Max Distance.
    rho_argsorted = np.argsort(rho)[::-1]
    max_rho = rho[rho_argsorted[0]]
    max_dist = np.max(distances_)

    if delta_cutoff == None:
        delta_cutoff = max_dist
    delta = [max_dist for i in xrange(X_len)]
    nneigh = [i for i in xrange(X_len)]

    # Calculating Delta.
    for i in xrange(0, X_len-1):
        for j in xrange(i, X_len):
            if rho[j] > rho[i]:
                delta[i] = distances_[i, j]
                nneigh[i] = j
                break

    # Clustering Data.
    # Initially, all samples are noise.

    ratio = delta * rho
    ratio_argsorted = np.argsort(ratio)[::-1]
    cut_index = ratio_argsorted[n_clusters-1]

    rho_cut = rho[cut_index]
    delta_cut = delta[cut_index]
    print "rho_cut:", rho_cut, cut_index
    print "delta_cut", delta_cut, cut_index

    labels_ = -np.ones(X_len, dtype=np.intp)
    cluster_centers_list = []
    cluster = 0

    for i in xrange(len(rho_argsorted)):
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


class density_peaks(BaseEstimator, ClusterMixin):
    """Adaptive Partitioning by Local Density-peaks (density_peaks)
    We present an efficient density-based adaptive-resolution clustering method density_peaks
    for analyzing large-scale molecular dynamics (MD) trajectories. density_peaks performs the
    k-nearest-neighbors search to estimate the density of MD conformations in a local fashion,
    which can group MD conformations in the same high-density region into a cluster.
    density_peaks greatly improves the popular density peaks algorithm by reducing the running
    time and the memory usage by 2-3 orders of magnitude for systems ranging from alanine
    dipeptide to a 370-residue Maltose-binding protein.  In addition, we demonstrate that
    density_peaks can produce clusters with various sizes that are adaptive to the underlying density
    (i.e. larger clusters at low density regions, while smaller clusters at high density regions),
    which is a clear advantage over other popular clustering algorithms including k-centers and
    k-medoids. We anticipate that density_peaks can be widely applied to split ultra-large MD datasets
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
    ensity-peaks (density_peaks): An efficient density-based clustering algorithm
    for analyzing molecular dynamics trajectories". Journal of Computational
    Chemistry XX/XX/2016: Vol.XXX No. XXX pp. XXXX-XXXX
    DOI: XXXX

    Alex Rodriguez, Alessandro Laio, "Clustering by fast search and find
    of density_peakss". Science 27 June 2014: Vol. 344 no. 6191 pp. 1492-1496
    DOI: 10.1126/science.1242072
    """

    def __init__(self, n_clusters=3, rho_cutoff=1.0, delta_cutoff=1.0, percent=0.2,
                 metric='rmsd', parallel=True):
        self.rho_cutoff = rho_cutoff
        self.delta_cutoff = delta_cutoff
        self.percent = percent
        self.metric = metric
        self.parallel = parallel
        self.n_clusters = n_clusters

    def fit(self, X, weight=None):
        """Perform clustering.
        Parameters
        -----------
        X : array-like, shape=[n_samples, n_features]
            Samples to cluster.
        """
        #        X = check_array(X)
        print "Doing density_peaks clustering..."
        t0 = time.time()
        self.cluster_centers_, self.labels_ = \
            density_peaks_clustering(X, n_clusters=self.n_clusters, rho_cutoff=self.rho_cutoff, delta_cutoff=self.delta_cutoff,
                             percent=self.percent, metric=self.metric, parallel=self.parallel)
        t1 = time.time()
        print "density_peaks Time Cost:", t1 - t0
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