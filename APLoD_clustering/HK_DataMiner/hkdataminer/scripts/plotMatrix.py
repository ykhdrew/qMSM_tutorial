__author__ = 'stephen'
import os,sys
import numpy as np
import scipy.io
HK_DataMiner_Path = os.path.relpath(os.pardir)
#HK_DataMiner_Path = os.path.abspath("/home/stephen/Dropbox/projects/work-2015.5/HK_DataMiner/")
sys.path.append(HK_DataMiner_Path)
#from utils import plot_matrix, plot_block_matrix
#from msm import MarkovStateModel
import argparse
from lumping import Evaluate_Result
cli = argparse.ArgumentParser()
cli.add_argument('-c', '--assignments_dir', type=str)
cli.add_argument('-m', '--microstate_mapping_dir', type=str, default=None)
cli.add_argument('-p', '--micro_tProb_dir', type=str, default=None)
cli.add_argument('-o', '--name', type=str, default='Matrix')
args = cli.parse_args()

assignments_dir = args.assignments_dir
microstate_mapping_dir = args.microstate_mapping_dir
micro_tProb_dir = args.micro_tProb_dir

name = args.name

labels = np.loadtxt(assignments_dir, dtype=np.int32)
if microstate_mapping_dir is not None:
    microstate_mapping = np.loadtxt(microstate_mapping_dir, dtype=np.int32)

if micro_tProb_dir is not None:
    micro_tProb_ = scipy.io.mmread(micro_tProb_dir)

Evaluate_Result(tProb_=micro_tProb_, lag_time=1, microstate_mapping_=microstate_mapping, MacroAssignments_=labels, name=name)

'''
MacroMSM = MarkovStateModel(lag_time=1)
MacroMSM.fit(labels)
tProb_ = MacroMSM.tProb_
plot_matrix(tProb_=tProb_, name=name)
metastability = tProb_.diagonal().sum()
metastability /= len(tProb_)
print "metastability:", metastability

#Begin modularity calculation
degree = micro_tProb_.sum(axis=1) #row sum of tProb_ matrix
total_degree = degree.sum()

modularity = 0.0
len_mapping = len(microstate_mapping)
for i in xrange(len_mapping):
    state_i = microstate_mapping[i]
    for j in xrange(len_mapping):
        state_j = microstate_mapping[j]
        if state_i == state_j:
            modularity += micro_tProb_[i, j] - degree[i]*degree[j]/total_degree
modularity /= total_degree
print "modularity:", modularity

'''

