__author__ = 'stephen'
import os,sys
import numpy as np
import scipy.io
#import matplotlib.pyplot as plt
from collections import Counter
HK_DataMiner_Path = os.path.relpath(os.pardir)
#HK_DataMiner_Path = os.path.abspath("/home/stephen/Dropbox/projects/work-2015.5/HK_DataMiner/")
sys.path.append(HK_DataMiner_Path)
import argparse
from utils import plot_cluster_size_distribution

cli = argparse.ArgumentParser()
cli.add_argument('-c', '--assignments_dir', type=str)
args = cli.parse_args()

assignments_dir = args.assignments_dir


labels = np.loadtxt(assignments_dir, dtype=np.int32)

#counts = Counter(labels)
counts = list(Counter(labels).values())
total_states = np.max(labels) + 1
states_magnitude = int(np.ceil(np.log10(total_states)))
total_frames = len(labels)
frames_magnitude = int(np.ceil(np.log10(total_frames)))
print "states", total_states, "frames", total_frames

populations = np.zeros(frames_magnitude+1)
for i in counts:
    if i > 0:
        log_i = np.log10(i)
        magnitude = np.ceil(log_i)
        populations[magnitude] += 1

#print magnitude populations
print "Populations Probability:"
#bins = [0]
for i in xrange(len(populations)):
    populations[i] = populations[i] / total_states
    print "10 ^", i, "to", "10 ^", i+1,":", populations[i]*100, "%"
    #bins.append(10**(i+1))

name = assignments_dir[:-4] + '_Populations'
print "name:", name
plot_cluster_size_distribution(populations=populations, name=name)

#print bins
#plt.hist(labels)
#plt.show()

