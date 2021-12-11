__author__ = 'stephen'
# ===============================================================================
# GLOBAL IMPORTS:
import os, sys
import time
from sklearn import cluster
# ===============================================================================
# LOCAL IMPORTS:
HK_DataMiner_Path = os.path.relpath(os.pardir)
sys.path.append(HK_DataMiner_Path)
from msm.msm import MarkovStateModel
from lumper_ import *
# ===============================================================================
class SpectralClustering(MarkovStateModel):
    def __init__(self, n_macro_states=None, lag_time=1, homedir=None, traj_len=None):
        # lag time in number of entries in assignment file (int).
        self.n_macro_states = n_macro_states
        self.assignments = None
        self.microstate_mapping_ = None
        self.MacroAssignments_ = None
        MarkovStateModel.__init__(self, lag_time=lag_time, n_macro_states=n_macro_states, traj_len=traj_len)

    def fit(self, assignments):
        t0=time.time()
        self.assignments = assignments
        super(SpectralClustering, self).fit(assignments)
        self.run()
        t1=time.time()
        print "Spectral Clustering Lumping running time:", t1-t0
        return self.MacroAssignments_

    def run(self):
        '''
        Do Spectral Lumping.
        '''
        print "Doing Spectral Clustering..."
        t0 = time.time()
        #spectral = cluster.SpectralClustering(n_clusters=self.n_macro_states, eigen_solver='arpack', affinity='precomputed', assign_labels='kmeans', n_init=10)
        spectral = cluster.SpectralClustering(n_clusters=self.n_macro_states, eigen_solver='arpack', affinity='precomputed', assign_labels='discretize')
        spectral.fit(self.tCount_)
        t1 = time.time()
        print "Time:", t1-t0
        #Spectral Clustering Results: microstate_mapping_
        self.microstate_mapping_ = spectral.labels_

        self.MacroAssignments_ = get_MacroAssignments(assignments=self.assignments,
                                                      microstate_mapping=self.microstate_mapping_)
        name = 'Spectral_' + str(self.n_micro_states) + '_to_' + str(self.n_macro_states) + '_states_'
        OutputResult(homedir=self.homedir, tCount_=self.tCount_, tProb_=self.tProb_,
                     microstate_mapping_=self.microstate_mapping_, MacroAssignments_=self.MacroAssignments_, name=name)