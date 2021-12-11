__author__ = 'stephen'
# ===============================================================================
# GLOBAL IMPORTS:
import os,sys
import numpy as np
import argparse
import mdtraj as md
# ===============================================================================
# LOCAL IMPORTS:
HK_DataMiner_Path = os.path.relpath(os.pardir)
#HK_DataMiner_Path = os.path.abspath("/home/stephen/Dropbox/projects/work-2015.5/HK_DataMiner/")
print HK_DataMiner_Path
sys.path.append(HK_DataMiner_Path)
from cluster import KCenters
from lumping import PCCA, PCCA_Standard, SpectralClustering, Ward
from utils import XTCReader, plot_cluster, utils, calculate_landscape, calculate_population 
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
cli.add_argument('-l', '--alignment', default=False, type=bool)

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
print "stride:", stride
trajreader = XTCReader(trajlistname, atom_indicesname, homedir, trajext, File_TOP, nSubSample=stride)
trajs = trajreader.trajs
traj_len = trajreader.traj_len
np.savetxt("./traj_len.txt", traj_len, fmt="%d")

# ===========================================================================
## get phi psi angles for Alanine Dipeptide
#if os.path.isfile("./phi_angles.txt") and os.path.isfile("./psi_angles.txt") is True:
#    phi_angles = np.loadtxt("./phi_angles.txt", dtype=np.float32)
#    psi_angles = np.loadtxt("./psi_angles.txt", dtype=np.float32)
#    phi_psi = np.column_stack((phi_angles, psi_angles))
#else:
#    phi_angles, psi_angles = trajreader.get_phipsi(trajs, psi=[6, 8, 14, 16], phi=[4, 6, 8, 14])
#    np.savetxt("./phi_angles.txt", phi_angles, fmt="%f")
#    np.savetxt("./psi_angles.txt", psi_angles, fmt="%f")

# ===========================================================================
# superpose
print "Alignment?", args.alignment
if args.alignment is True:
    align_atom_indices = np.loadtxt('align_atom_indices', dtype=np.int32).tolist()
    print "align_atom_indices:", align_atom_indices
    trajs.superpose(reference=trajs[0], frame=0, atom_indices=align_atom_indices)
    print "Alignment done."
# ===========================================================================

# ===========================================================================
# Just keep the atoms in atom indices, remove other atoms
atom_indices = np.loadtxt('atom_indices', dtype=np.int32).tolist()
print "atom_indices:", atom_indices
trajs_sub_atoms = trajs.atom_slice(atom_indices, inplace=False) #just keep the the atoms in atom indices
print "Trajs:", trajs
print "Sub_atoms_trajs:", trajs_sub_atoms

# ===========================================================================
# Reading Clustering Centers
centers = md.load("cluster_centers_sub_atoms.pdb")
print "Centers:", centers
# ===========================================================================
# do Assigning using KCenters method
#cluster = KCenters(n_clusters=n_clusters, metric="euclidean", random_state=0)
cluster = KCenters(centers=centers, n_clusters=n_clusters, metric="rmsd", random_state=0)
print cluster
#cluster.fit(phi_psi)
#cluster.fit(trajs_sub_atoms)
cluster.assign(trajs_sub_atoms, cluster)

labels = cluster.labels_
print labels
n_microstates = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_microstates)

# plot micro states
clustering_name = "kcenters_assign_n_" + str(n_microstates)
np.savetxt("assignments_"+clustering_name+".txt", labels, fmt="%d")


#plot_cluster(labels=labels, phi_angles=phi_angles, psi_angles=psi_angles, name=clustering_name)
#calculate_landscape(labels=labels, centers=cluster_centers_, phi_angles=phi_angles, psi_angles=psi_angles, potential=False, name=clustering_name)
#calculate_population(labels=labels, name=clustering_name)
