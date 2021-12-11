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
cli.add_argument('-c', '--assignments',  type=str)
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

diheds_array = np.loadtxt('diheds.txt', dtype=np.float32)
phi_angles = diheds_array[:,0]
psi_angles = diheds_array[:,1]
print(phi_angles.shape)

n_size = X.shape[0]
dimension = X.shape[1]


labels = np.loadtxt(args.assignments, dtype=np.int32)

#print(sk_labels)
n_microstates = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_microstates)

# plot micro states
name =args.assignments.split("/")[-1][:-4]
clustering_name = "ML-DBSCAN_n_" + name + "_"+ str(n_microstates)
#np.savetxt("assignments.txt", labels, fmt="%d")
##np.savetxt("cluster_centers_"+clustering_name+".txt", cluster_centers_, fmt="%d")
plot_cluster(labels=labels, phi_angles=phi_angles, psi_angles=psi_angles, name=clustering_name+'_rama', potential=False)

plot_cluster(labels=labels, phi_angles=X[:, 0], psi_angles=X[:, 1], name=clustering_name+'_tica', potential=False)
