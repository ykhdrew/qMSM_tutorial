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
from utils import plot_cluster, XTCReader, VectorReader
# ===============================================================================
cli = argparse.ArgumentParser()
cli.add_argument('-t',   '--trajListFns', default ='tica_trajlist',
                 help='List of trajectory files to read in, separated by spaces.')
cli.add_argument('-o',   '--homedir',  help='Home dir.', default=".", type=str)
cli.add_argument('-d',   '--device', help='Device No. of GPU.', default=0, type=int)
#cli.add_argument('-n',   '--n_size', help='database size.', default=10000, type=int)
cli.add_argument('-e',   '--eps', help='eps', default=1, type=float)
cli.add_argument('-m',   '--min_samples', help='min_samples', default=5, type=int)
cli.add_argument('-l',   '--nlist', help='nlist', default=1000, type=int)
cli.add_argument('-p',   '--nprobe', help='nprob', default=10, type=int)
cli.add_argument('-s',   '--stride', help='Subsample stride', default = None, type=int)
# ===========================================================================
args = cli.parse_args()
trajlistName = args.trajListFns
homedir = args.homedir
device = args.device
#dimension = args.dimension  # dimension
#n_size = args.n_size     # database size
#np.random.seed(1234)             # make reproducible
#X = np.random.random((n_size, dimension)).astype('float32') * 10.0
#X[:, 0] += np.arange(n_size) / 100.0
#print(X)

#if args.stride is not None:
#    trajreader = VectorReader(trajlistName=trajlistName, homedir=homedir, trajExt='txt', stride=args.stride)
#else:
#    trajreader = VectorReader(trajlistName=trajlistName, homedir=homedir, trajExt='txt')
#X = trajreader.trajs

X = []
for line in open('trajlist_ala'):
    X.append(np.loadtxt(line.strip()))

X = np.concatenate(X)
print(X)
#traj_len = trajreader.traj_len
#np.savetxt("./traj_len.txt", traj_len, fmt="%d")

'''
trajreader = XTCReader('trajlist', 'atom_indices', '.', 'xtc', 'native.pdb', nSubSample=None)
trajs = trajreader.trajs
traj_len = trajreader.traj_len
xyz = trajs
from msmbuilder.featurizer import DihedralFeaturizer

featurizer = DihedralFeaturizer(types=['phi', 'psi'], sincos=False)
diheds_list = featurizer.fit_transform(xyz)
diheds = diheds_list[0].tolist()
for i in range(1, len(diheds_list)):
    diheds.append(diheds_list[i][0].tolist())
diheds_array = np.asarray(diheds) * 180.0 / np.pi
print(diheds_array)

#print(diheds_list)
#print(diheds)
np.savetxt('diheds.txt', diheds_array, fmt="%8f")

#with open('diheds.txt', 'w') as f:
#    for item in diheds:
#        f.write("%s\n" % item)

print(diheds_array.shape)
'''
diheds_array = np.loadtxt('diheds.txt', dtype=np.float32)
phi_angles = diheds_array[:,0]
psi_angles = diheds_array[:,1]
print(phi_angles.shape)
# ===========================================================================
#if os.path.isfile("./phi_angles.txt") and os.path.isfile("./psi_angles.txt") is True:
#    phi_angles = np.loadtxt("./phi_angles.txt", dtype=np.float32)
#    psi_angles = np.loadtxt("./psi_angles.txt", dtype=np.float32)
#X = np.column_stack((phi_angles, psi_angles))
# print(X.shape)
#phi_angles = np.degrees(diheds[:, 0])
#psi_angles = np.degrees(diheds[:, 1])
#print(phi_angles)
#X = diheds_array
#X = tica_trajs[0].astype(np.float32)
# print(X)
n_size = X.shape[0]
dimension = X.shape[1]
'''
#if os.path.isfile("./phi_angles.txt") and os.path.isfile("./psi_angles.txt") is True:
#    phi_angles = np.loadtxt("./phi_angles.txt", dtype=np.float32)
#    psi_angles = np.loadtxt("./psi_angles.txt", dtype=np.float32)

#X=np.column_stack((phi_angles, psi_angles))
#print(X.shape)
#n_size = X.shape[0]
#dimension = X.shape[1]
'''
# ===========================================================================
eps = args.eps  # eps
min_samples = args.min_samples  # min_samples
nlist = args.nlist
nprobe = args.nprobe
IVFFlat = True
print('n_size = %d,\t dimension = %d,\t eps = %f, min_samples = %d' % ( n_size, dimension, eps, min_samples))
'''
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
plot_cluster(labels=CPU_IVFFlat_labels, phi_angles=phi_angles, psi_angles=psi_angles, name=clustering_name, potential=False)
np.savetxt("assignments_"+clustering_name+".txt", CPU_IVFFlat_labels , fmt="%d")

'''
# ===========================================================================
# do Clustering using Scikit-Learn DBSCAN method
#from sklearn.cluster import DBSCAN

from cluster import DBSCAN
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
np.savetxt("assignments.txt", sk_labels, fmt="%d")
#np.savetxt("cluster_centers_"+clustering_name+".txt", cluster_centers_, fmt="%d")
plot_cluster(labels=sk_labels, phi_angles=phi_angles, psi_angles=psi_angles, name='rama', potential=False)

plot_cluster(labels=sk_labels, phi_angles=X[:, 0], psi_angles=X[:, 1], name='tica', potential=False)

print('---------------------------------------------------------------------------------')
##print('%f\t%f' % (CPU_time, Sklearn_time))
#print('%f\t%f\t%f' % (GPU_time, CPU_time, Sklearn_time))
print('---------------------------------------------------------------------------------')