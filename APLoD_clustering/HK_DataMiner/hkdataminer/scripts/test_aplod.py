__author__ = 'stephen'
import os,sys
import numpy as np
HK_DataMiner_Path = os.path.relpath(os.pardir)
#HK_DataMiner_Path = os.path.abspath("/Users/stephen/projects_local/xiaowei-aplod/xiaowei-aplod/HK_DataMiner-master/hkdataminer")
sys.path.append(HK_DataMiner_Path)
from cluster import APLoD, run_knn
#from lumping import PCCA, PCCA_Standard, SpectralClustering, Ward
from utils import VectorReader, plot_cluster
import argparse
import time
from functools import reduce

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
cli.add_argument('-n',   '--n_clusters', help='n_clusters.',
                 default=100, type=int)
cli.add_argument('-m',   '--n_macro_states', help='n_macro_states.',
                 default=6, type=int)
cli.add_argument('-s',   '--stride', help='stride.',
                 default=None, type=int)
cli.add_argument('-b',   '--n_neighbors', help='n_neighbors.',
                 default=10, type=int)
cli.add_argument('-r',   '--rho_cutoff', help='Density Peak rho cutoff.',
                 default=1.0, type=float)
cli.add_argument('-d',   '--delta_cutoff', help='Density Peak delta cutoff.',
                 default=1.0, type=float)
cli.add_argument('-l',   '--lumping_method', help='lumping_methd.',
                 default='PCCA', type=str)
args = cli.parse_args()
trajlistname = args.trajListFns
atom_indicesname = args.atomListFns
trajext = args.iext
File_TOP = args.topology
homedir = args.homedir
n_clusters = args.n_clusters
n_macro_states = args.n_macro_states
n_neighbors = args.n_neighbors
rho_cutoff = args.rho_cutoff
delta_cutoff = args.delta_cutoff
stride = args.stride
#===========================================================================
#Reading Trajs from Vector files
print("stride:", stride)
trajreader = VectorReader(trajlistName='trajlist', homedir='./', trajExt='txt')
trajs = trajreader.trajs
#trajs.center_coordinates()
traj_len = trajreader.traj_len
np.savetxt("./traj_len.txt", traj_len, fmt="%d")
#trajs.save("villin_trajs.pdb")


# ===========================================================================
# do Clustering using Density Peak method
# Initializing Density Peak algorithm
#algorithm = "precomputed"
algorithm = "kd_tree"
if algorithm is "precomputed":
    #sample_dist_metric, distances_, indices = run_knn(X=trajs, n_neighbors=n_neighbors, metric="rmsd")
    sample_dist_metric = np.loadtxt("./sample_dist_metric.txt", dtype=np.float32)
    distances_ = np.loadtxt("./distances_.txt", dtype=np.float32)
    indices = np.loadtxt("./indices.txt", dtype=np.int32)
else:
    sample_dist_metric = None
    distances_ = None
    indices = None

print("delta_cutoff:", delta_cutoff, "rho_cutoff:", rho_cutoff)
cluster = APLoD(rho_cutoff=rho_cutoff, delta_cutoff=delta_cutoff, n_neighbors=n_neighbors,
                      metric="euclidean", algorithm=algorithm, sample_dist_metric=sample_dist_metric,
                      distances_=distances_, indices=indices)
cluster.fit(trajs)

# get the assignments of micro states
assignments_dp = cluster.labels_
cluster_centers_dp = cluster.cluster_centers_
print(assignments_dp)

# check out how many microstates
n_microstates = len(set(assignments_dp)) - (1 if -1 in assignments_dp else 0)
print('Estimated number of clusters: %d' % n_microstates)
# plot micro states
clustering_name = "density_peak_n_" + str(n_microstates) + "_neighbors_" + str(n_neighbors) \
                + "_rho_" + str(rho_cutoff) + "_delta_" + str(delta_cutoff)
np.savetxt("assignments_"+clustering_name+".txt", assignments_dp, fmt="%d")
np.savetxt("cluster_centers_"+clustering_name+".txt", cluster_centers_dp, fmt="%d")
#plot_cluster(labels=assignments_dp, phi_angles=phi_angles, psi_angles=psi_angles, name=clustering_name)
#calculate_population(labels=labels, name=clustering_name)
