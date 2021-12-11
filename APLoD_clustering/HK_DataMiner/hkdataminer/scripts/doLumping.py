__author__ = 'stephen'
import os,sys
import numpy as np
HK_DataMiner_Path = os.path.relpath(os.pardir)
#HK_DataMiner_Path = os.path.abspath("/home/stephen/Dropbox/projects/work-2015.5/HK_DataMiner/")
sys.path.append(HK_DataMiner_Path)
from lumping import PCCA, PCCA_Standard, SpectralClustering, Ward, PCCA3, PCCA_Plus
from utils import plot_cluster, plot_each_cluster, XTCReader
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
cli.add_argument('-n',   '--n_clusters', help='''n_clusters.''',
                 default=100, type=int)
cli.add_argument('-m',   '--n_macro_states', help='''n__macro_states.''',
                 default=6, type=int)
cli.add_argument('-c', '--assignments', type=str)
cli.add_argument('-l', '--traj_len', type=str, default='traj_len.txt')
args = cli.parse_args()
trajlistname = args.trajListFns
atom_indicesname = args.atomListFns
trajext = args.iext
File_TOP = args.topology
homedir = args.homedir
n_clusters = args.n_clusters
n_macro_states = args.n_macro_states

# ===========================================================================
# Reading phi angles and psi angles data from XTC files
if os.path.isfile("./phi_angles.txt") and os.path.isfile("./psi_angles.txt") is True:
    phi_angles = np.loadtxt("./phi_angles.txt", dtype=np.float32)
    psi_angles = np.loadtxt("./psi_angles.txt", dtype=np.float32)
    phi_psi = np.column_stack((phi_angles, psi_angles))
else:
    trajreader = XTCReader(trajlistname, atom_indicesname, homedir, trajext, File_TOP)
    trajs = trajreader.trajs
    traj_len = trajreader.traj_len
    np.savetxt("./traj_len.txt", traj_len, fmt="%d")
    phi_angles, psi_angles = trajreader.get_phipsi(trajs, psi=[6, 8, 14, 16], phi=[4, 6, 8, 14])
    phi_psi = np.column_stack((phi_angles, psi_angles))
    np.savetxt("./phi_angles.txt", phi_angles, fmt="%f")
    np.savetxt("./psi_angles.txt", psi_angles, fmt="%f")
# ===========================================================================
# Reading split assignments and the length of each traj
assignments_dir = args.assignments
labels = np.loadtxt(assignments_dir, dtype=np.int32)

if os.path.isfile(args.traj_len) is True:
    traj_len = np.loadtxt(args.traj_len, dtype=np.int32)
else:
    trajreader = XTCReader(trajlistname, atom_indicesname, homedir, trajext, File_TOP)
    trajs = trajreader.trajs
    traj_len = trajreader.traj_len
    np.savetxt("./traj_len.txt", traj_len, fmt="%d")

n_microstates = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_microstates)

# ============================================================================
# Do Lumping

'''lumper_name=['PCCA', 'PCCA_mean', 'Spectral', 'Ward']
PCCA_lumper = PCCA(n_macro_states=n_macro_states, traj_len=traj_len, cut_by_mean=False)
PCCA_mean_lumper = PCCA(n_macro_states=n_macro_states, traj_len=traj_len, cut_by_mean=True)
PCCA_Standard_lumper = PCCA_Standard(n_macro_states=n_macro_states, traj_len=traj_len)
Spectral_lumper = SpectralClustering(n_macro_states=n_macro_states, traj_len=traj_len)
Ward_lumper = Ward(n_macro_states=n_macro_states, traj_len=traj_len)
lumper_method =[PCCA_lumper, PCCA_mean_lumper, Spectral_lumper, Ward_lumper]

for name, algorithm in zip(lumper_name, lumper_method):
    algorithm.fit(labels)
    lumping_name = 'micro_' + str(n_microstates) + "_" + name + "_" + str(n_macro_states)
    #macro_states =
    plot_cluster(labels=algorithm.MacroAssignments_, phi_angles=phi_angles, psi_angles=psi_angles, name=lumping_name)
'''


#print "Lumping data using PCCA Plus Algorithm"
#PCCA3_lumper = PCCA_Plus(n_macro_states)
#PCCA3_lumper.fit(labels, 200)
#MacroAssignments_ = PCCA3_lumper.transform(labels)
#lumping_name = 'micro_' + str(n_microstates) + "_" + "PCCA_Plus" + "_" + str(n_macro_states)
#plot_cluster(labels=MacroAssignments_, phi_angles=phi_angles, psi_angles=psi_angles, name=lumping_name)

#print "Lumping data using Ward Algorithm"
#lumper = Ward(n_macro_states=n_macro_states, traj_len=traj_len)
#lumper.fit(labels)
#lumping_name = 'micro_' + str(n_microstates) + "_" + "Ward" + "_" + str(n_macro_states)
#plot_cluster(labels=lumper.MacroAssignments_, phi_angles=phi_angles, psi_angles=psi_angles, name=lumping_name)


#for n_macro_states in xrange(:
print "n_macro_states:", n_macro_states
PCCA_lumper = PCCA(n_macro_states=n_macro_states, traj_len=traj_len, cut_by_mean=False)
algorithm = PCCA_lumper
name = 'PCCA'
algorithm.fit(labels)
lumping_name = 'micro_' + str(n_microstates) + "_" + name + "_" + str(n_macro_states)

#plot_cluster(labels=algorithm.MacroAssignments_, phi_angles=phi_angles, psi_angles=psi_angles, name=lumping_name)
#plot_each_cluster(labels=algorithm.MacroAssignments_, phi_angles=phi_angles, psi_angles=psi_angles, name=lumping_name)

#contour_cluster(labels=algorithm.MacroAssignments_, phi_angles=phi_angles, psi_angles=psi_angles, name=lumping_name)

 
