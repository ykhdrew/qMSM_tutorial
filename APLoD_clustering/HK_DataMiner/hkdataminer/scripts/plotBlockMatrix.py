__author__ = 'stephen'
import os,sys
import numpy as np
import scipy.io
HK_DataMiner_Path = os.path.relpath(os.pardir)
#HK_DataMiner_Path = os.path.abspath("/home/stephen/Dropbox/projects/work-2015.5/HK_DataMiner/")
sys.path.append(HK_DataMiner_Path)
from utils import plot_matrix, plot_block_matrix
from msm import MarkovStateModel
import argparse
cli = argparse.ArgumentParser()
cli.add_argument('-c', '--assignments_dir', type=str)
cli.add_argument('-p', '--tProb', type=str, default=None)
args = cli.parse_args()

assignments_dir = args.assignments_dir
tProb_dir = args.tProb

labels = np.loadtxt(assignments_dir, dtype=np.int32)

if tProb_dir is not None:
    tProb_ = scipy.io.mmread(tProb_dir)
    #plot_matrix(labels=labels, tProb_=tProb_, name='Matrix')
    plot_block_matrix(labels=labels, tProb_=tProb_, name='BlockMatrix')
else:
    MSM = MarkovStateModel(lag_time=1)
    MSM.fit(labels)
    tProb_ = MSM.tProb_
    #plot_matrix(labels=None, tProb_=tProb_, name='Matrix')
    plot_block_matrix(labels=labels, tProb_=tProb_, name='BlockMatrix')

