__author__ = 'stephen'
import os,sys
import numpy as np
HK_DataMiner_Path = os.path.relpath(os.pardir)
#HK_DataMiner_Path = os.path.abspath("/home/stephen/Dropbox/projects/work-2015.5/HK_DataMiner/")
sys.path.append(HK_DataMiner_Path)
from utils import plot_cluster, plot_each_cluster, XTCReader, plot_landscape
import argparse
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
cli.add_argument('-c', '--assignments', type=str)
cli.add_argument('-s', '--centers', type=str)
cli.add_argument('-l', '--traj_len', type=str, default='traj_len.txt')
args = cli.parse_args()
trajlistname = args.trajListFns
atom_indicesname = args.atomListFns
trajext = args.iext
File_TOP = args.topology
homedir = args.homedir

# ===========================================================================
# Reading phi angles and psi angles data from XTC files
if os.path.isfile("./phi_angles.txt") and os.path.isfile("./psi_angles.txt") is True:
    phi_angles = np.loadtxt("./phi_angles.txt", dtype=np.float32)
    psi_angles = np.loadtxt("./psi_angles.txt", dtype=np.float32)
else:
    trajreader = XTCReader(trajlistname, atom_indicesname, homedir, trajext, File_TOP)
    trajs = trajreader.trajs
    traj_len = trajreader.traj_len
    np.savetxt("./traj_len.txt", traj_len, fmt="%d")
    phi_angles, psi_angles = trajreader.get_phipsi(trajs, psi=[6, 8, 14, 16], phi=[4, 6, 8, 14])
    np.savetxt("./phi_angles.txt", phi_angles, fmt="%f")
    np.savetxt("./psi_angles.txt", psi_angles, fmt="%f")
# ===========================================================================
# Reading split assignments and the length of each traj
assignments_dir = args.assignments
labels = np.loadtxt(assignments_dir, dtype=np.int32)
#centers_dir = args.centers
#centers = np.loadtxt(centers_dir, dtype=np.int32)
#plot_cluster(labels=labels, phi_angles=phi_angles, psi_angles=psi_angles, name=dir+'Dihedrals'+assignments_dir)

#phi_ctr = phi_angles[centers]
#psi_ctr = psi_angles[centers]
#labels_ctr = labels[centers]
#plot_cluster(labels=labels_lf, phi_angles=phi_new, psi_angles=psi_new, name=clustering_name)

name = assignments_dir[:-4] + '_Energy_Landscape'
#print "name:", name
#plot_cluster_size_distribution(populations=populations, name=name)
plot_landscape(labels=None, phi_angles=phi_angles, psi_angles=psi_angles, phi_ctr=None, psi_ctr=None, name=name)
#plot_landscape(labels=None, phi_angles=phi_angles, psi_angles=psi_angles)
