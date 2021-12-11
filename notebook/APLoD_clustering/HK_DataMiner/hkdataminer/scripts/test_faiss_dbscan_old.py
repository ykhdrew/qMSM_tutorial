__author__ = 'stephen'
# ===============================================================================
# GLOBAL IMPORTS:
import os,sys
import numpy as np
import argparse
import time
# ===============================================================================
# LOCAL IMPORTS:
#HK_DataMiner_Path = os.path.relpath(os.pardir)
HK_DataMiner_Path = os.path.abspath("/Users/stephen/Dropbox/projects/work-2018.12/HK_DataMiner/")
#print HK_DataMiner_Path
sys.path.append(HK_DataMiner_Path)
from cluster import Faiss_DBSCAN
from utils import XTCReader, plot_cluster
# ===============================================================================
cli = argparse.ArgumentParser()
cli.add_argument('-t',   '--trajListFns', default = 'trajlist',
                 help='List of trajectory files to read in, separated by spaces.')
cli.add_argument('-a', '--atomListFns', default='atom_indices',
                 help='List of atom index files to read in, separated by spaces.')
cli.add_argument('-g',   '--topology', default='native.pdb', help='topology file.')
cli.add_argument('-o',   '--homedir',  help='Home dir.', default=".", type=str)
cli.add_argument('-e',   '--iext', help='''The file extension of input trajectory
                 files.  Must be a filetype that mdtraj.load() can recognize.''',
                 default="xtc", type=str)
cli.add_argument('-n',   '--n_clusters', help='''n_clusters.''',
                 default=100, type=int)
cli.add_argument('-m',   '--n_macro_states', help='''n_macro_states.''',
                 default=6, type=int)
cli.add_argument('-s',   '--stride', help='stride.',
                 default=None, type=int)

args = cli.parse_args()
trajlistname = args.trajListFns
atom_indicesname = args.atomListFns
trajext = args.iext
File_TOP = args.topology
homedir = args.homedir
n_clusters = args.n_clusters
n_macro_states = args.n_macro_states
stride = args.stride
# ===========================================================================
# Reading Trajs from XTC files
#print "stride:", stride
#trajreader = XTCReader(trajlistname, atom_indicesname, homedir, trajext, File_TOP, nSubSample=stride)
#trajs = trajreader.trajs
#print(trajs)
#traj_len = trajreader.traj_len
#np.savetxt("./traj_len.txt", traj_len, fmt="%d")

if os.path.isfile("./phi_angles.txt") and os.path.isfile("./psi_angles.txt") is True:
    phi_angles = np.loadtxt("./phi_angles.txt", dtype=np.float32)
    psi_angles = np.loadtxt("./psi_angles.txt", dtype=np.float32)
else:
    phi_angles, psi_angles = trajreader.get_phipsi(trajs, psi=[6, 8, 14, 16], phi=[4, 6, 8, 14])
    #phi_angles, psi_angles = trajreader.get_phipsi(trajs, psi=[5, 7, 13, 15], phi=[3, 5, 7, 13])
    np.savetxt("./phi_angles.txt", phi_angles, fmt="%f")
    np.savetxt("./psi_angles.txt", psi_angles, fmt="%f")

phi_psi=np.column_stack((phi_angles, psi_angles))
print(phi_psi.shape)
# ===========================================================================
eps = 30.0
min_samples = 5
# ===========================================================================
# do Clustering using Faiss GPU DBSCAN method
'''
GPU_cluster = Faiss_DBSCAN(eps=eps, min_samples=min_samples, metric="l2", GPU=False)
print(GPU_cluster)
GPU_cluster.fit(phi_psi)
#cluster.fit(trajs)

GPU_labels = GPU_cluster.labels_
print(GPU_labels)
n_microstates = len(set(GPU_labels)) - (1 if -1 in GPU_labels else 0)
print('Estimated number of clusters: %d' % n_microstates)

#cluster_centers_ = cluster.cluster_centers_
# plot micro states
clustering_name = "GPU_Faiss_dbscan_n_" + str(n_microstates)
np.savetxt("assignments_"+clustering_name+".txt", GPU_labels, fmt="%d")
#np.savetxt("cluster_centers_"+clustering_name+".txt", cluster_centers_, fmt="%d")

plot_cluster(labels=GPU_labels, phi_angles=phi_angles, psi_angles=psi_angles, name=clustering_name)
'''
# ===========================================================================
# do Clustering using Faiss CPU DBSCAN method
CPU_IVFFlat_cluster = Faiss_DBSCAN(eps=eps, min_samples=min_samples,nlist=100, nprobe=5, metric="l2", GPU=False, IVFFlat=True)
print(CPU_IVFFlat_cluster)
CPU_IVFFlat_cluster.fit(phi_psi)
#cluster.fit(trajs)

CPU_IVFFlat_labels = CPU_IVFFlat_cluster.labels_
print(CPU_IVFFlat_labels)
n_microstates = len(set(CPU_IVFFlat_labels)) - (1 if -1 in CPU_IVFFlat_labels else 0)
print('Estimated number of clusters: %d' % n_microstates)

#cluster_centers_ = cluster.cluster_centers_
# plot micro states
clustering_name = "CPU_Faiss_IVFFlat_dbscan_n_" + str(n_microstates)
np.savetxt("assignments_"+clustering_name+".txt", CPU_IVFFlat_labels, fmt="%d")
#np.savetxt("cluster_centers_"+clustering_name+".txt", cluster_centers_, fmt="%d")

plot_cluster(labels=CPU_IVFFlat_labels, phi_angles=phi_angles, psi_angles=psi_angles, name=clustering_name)
# ===========================================================================
# do Clustering using Faiss CPU DBSCAN method
CPU_FlatL2_cluster = Faiss_DBSCAN(eps=eps, min_samples=min_samples,nlist=100, nprobe=5, metric="l2", GPU=False, IVFFlat=False)
print(CPU_FlatL2_cluster)
CPU_FlatL2_cluster.fit(phi_psi)
#cluster.fit(trajs)

CPU_FlatL2_labels = CPU_FlatL2_cluster.labels_
print(CPU_FlatL2_labels)
n_microstates = len(set(CPU_FlatL2_labels)) - (1 if -1 in CPU_FlatL2_labels else 0)
print('Estimated number of clusters: %d' % n_microstates)

#cluster_centers_ = cluster.cluster_centers_
# plot micro states
clustering_name = "CPU_Faiss_FlatL2_dbscan_n_" + str(n_microstates)
np.savetxt("assignments_"+clustering_name+".txt", CPU_FlatL2_labels, fmt="%d")
#np.savetxt("cluster_centers_"+clustering_name+".txt", cluster_centers_, fmt="%d")

plot_cluster(labels=CPU_FlatL2_labels, phi_angles=phi_angles, psi_angles=psi_angles, name=clustering_name)
# ===========================================================================
# do Clustering using Scikit-Learn DBSCAN method
from sklearn.cluster import DBSCAN
sk_cluster = DBSCAN(eps=eps, min_samples=min_samples, metric="l2")
t0 = time.time()
sk_cluster.fit(phi_psi)
t1 = time.time()
print("Scikit-Learn DBSCAN clustering Time Cost:", t1 - t0)

sk_labels = sk_cluster.labels_
print(sk_labels)
n_microstates = len(set(sk_labels)) - (1 if -1 in sk_labels else 0)
print('Estimated number of clusters: %d' % n_microstates)

#cluster_centers_ = cluster.cluster_centers_
# plot micro states
clustering_name = "Sklearn_dbscan_n_" + str(n_microstates)
np.savetxt("assignments_"+clustering_name+".txt", sk_labels, fmt="%d")
#np.savetxt("cluster_centers_"+clustering_name+".txt", cluster_centers_, fmt="%d")

plot_cluster(labels=sk_labels, phi_angles=phi_angles, psi_angles=psi_angles, name=clustering_name)