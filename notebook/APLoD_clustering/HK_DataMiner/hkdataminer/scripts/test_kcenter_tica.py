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
from cluster import Faiss_DBSCAN, KCenters
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
n_clusters = 10
print('---------------------------------------------------------------------------------')

from msmbuilder.cluster import KCenters
cluster = KCenters(n_clusters=n_clusters, metric="euclidean", random_state=0)
print(cluster)
#cluster.fit(phi_psi)

cluster.fit(X)
labels = cluster.labels_
print(labels)

labels = np.concatenate(labels)
n_microstates = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_microstates)

#cluster_centers_ = cluster.cluster_centers_
# plot micro states
clustering_name = "kcenters_n_" + str(n_microstates)
#splited_assignments =split_assignments(labels, traj_len)
#np.savetxt("assignments_"+clustering_name+".txt", labels, fmt="%d")
np.savetxt("assignments_"+clustering_name+".txt", labels , fmt="%d")
#np.savetxt("cluster_centers_"+clustering_name+".txt", cluster_centers_, fmt="%d")
plot_cluster(labels=labels, phi_angles=phi_angles, psi_angles=psi_angles, name=clustering_name)

X = np.concatenate(X)
plot_cluster(labels=labels, phi_angles=X[:, 0], psi_angles=X[:, 1], name='tica_clustering.png')

#trajs[cluster_centers_].save("cluster_centers.pdb")
#trajs_sub_atoms[cluster_centers_].save("cluster_centers_sub_atoms.pdb")
