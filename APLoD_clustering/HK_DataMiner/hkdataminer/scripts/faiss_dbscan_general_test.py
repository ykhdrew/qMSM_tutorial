__author__ = 'stephen'
# ===============================================================================
# GLOBAL IMPORTS:
import os,sys
import numpy as np
import argparse
import time
# ===============================================================================
# LOCAL IMPORTS:
HK_DataMiner_Path = os.path.relpath(os.pardir)
#HK_DataMiner_Path = os.path.abspath("/Users/stephen/Dropbox/projects/work-2018.12/HK_DataMiner/")
#print HK_DataMiner_Path
sys.path.append(HK_DataMiner_Path)
from cluster import Faiss_DBSCAN
# ===============================================================================
cli = argparse.ArgumentParser()
cli.add_argument('-d',   '--dimension', help='dimension of the database.', default=2, type=int)
cli.add_argument('-n',   '--n_size', help='database size.', default=10000, type=int)
cli.add_argument('-e',   '--eps', help='eps', default=1, type=float)
cli.add_argument('-m',   '--min_samples', help='min_samples', default=5, type=int)
cli.add_argument('-l',   '--nlist', help='nlist', default=1000, type=int)
cli.add_argument('-p',   '--nprobe', help='nprob', default=10, type=int)
# ===========================================================================
args = cli.parse_args()
dimension = args.dimension  # dimension
n_size = args.n_size     # database size
np.random.seed(1234)             # make reproducible
X = np.random.random((n_size, dimension)).astype('float32') * 10.0
X[:, 0] += np.arange(n_size) / 100.0
#print(X)
# ===========================================================================
eps = args.eps  # eps
min_samples = args.min_samples  # min_samples
nlist = args.nlist
nprobe = args.nprobe
IVFFlat = True
print('n_size = %d,\t dimension = %d,\t eps = %f, min_samples = %d' % ( n_size, dimension, eps, min_samples))
# ===========================================================================
# do Clustering using Faiss GPU DBSCAN based on IVFFlat

GPU_IVFFlat_cluster  = Faiss_DBSCAN(eps=eps, min_samples=min_samples, nlist=nlist, nprobe=nprobe, metric="l2", GPU=True, IVFFlat=IVFFlat)
print(GPU_IVFFlat_cluster)
t0 = time.time()
GPU_IVFFlat_cluster.fit(X)
t1 = time.time()
GPU_time = t1 - t0
print("Clustering using Faiss GPU DBSCAN based on IVFFlat Time Cost:", t1 - t0)

GPU_IVFFlat_labels = GPU_IVFFlat_cluster.labels_
#print(GPU_IVFFlat_labels)
n_microstates = len(set(GPU_IVFFlat_labels)) - (1 if -1 in GPU_IVFFlat_labels else 0)
print('Estimated number of clusters: %d' % n_microstates)

# plot micro states
clustering_name = "GPU_Faiss_IVFFlat_dbscan_n_" + str(n_microstates)

# ===========================================================================
# do Clustering using Faiss CPU DBSCAN based on IVFFlat

CPU_IVFFlat_cluster = Faiss_DBSCAN(eps=eps, min_samples=min_samples, nlist=nlist, nprobe=nprobe, metric="l2", GPU=False, IVFFlat=IVFFlat)
print(CPU_IVFFlat_cluster)
t0 = time.time()
CPU_IVFFlat_cluster.fit(X)
t1 = time.time()
CPU_time = t1 - t0
print("Clustering using Faiss CPU DBSCAN based on IVFFlat Time Cost:", t1 - t0)

CPU_IVFFlat_labels = CPU_IVFFlat_cluster.labels_
#print(CPU_IVFFlat_labels)
n_microstates = len(set(CPU_IVFFlat_labels)) - (1 if -1 in CPU_IVFFlat_labels else 0)
print('Estimated number of clusters: %d' % n_microstates)

# plot micro states
clustering_name = "CPU_Faiss_IVFFlat_dbscan_n_" + str(n_microstates)

# ===========================================================================
# do Clustering using Scikit-Learn DBSCAN method

from sklearn.cluster import DBSCAN
sk_cluster = DBSCAN(eps=eps, min_samples=min_samples, metric="l2")
t0 = time.time()
sk_cluster.fit(X)
t1 = time.time()
Sklearn_time = t1 - t0
print("Clustering using Scikit-Learn DBSCAN Time Cost:", t1 - t0)

sk_labels = sk_cluster.labels_
#print(sk_labels)
n_microstates = len(set(sk_labels)) - (1 if -1 in sk_labels else 0)
print('Estimated number of clusters: %d' % n_microstates)

# plot micro states
clustering_name = "Sklearn_dbscan_n_" + str(n_microstates)
#np.savetxt("assignments_"+clustering_name+".txt", sk_labels, fmt="%d")
#np.savetxt("cluster_centers_"+clustering_name+".txt", cluster_centers_, fmt="%d")
print('---------------------------------------------------------------------------------')
print('%f\t%f\t%f' % (GPU_time, CPU_time, Sklearn_time))
print('---------------------------------------------------------------------------------')


