__author__ = 'stephen'
# ===============================================================================
# GLOBAL IMPORTS:
import os, sys
import time
from sklearn import cluster
from sklearn.neighbors import kneighbors_graph
# ===============================================================================
# LOCAL IMPORTS:
HK_DataMiner_Path = os.path.relpath(os.pardir)
sys.path.append(HK_DataMiner_Path)
from msm.msm import MarkovStateModel
from lumper_ import *
# ===============================================================================
class Ward(MarkovStateModel):
    def __init__(self, n_macro_states=None, lag_time=1, homedir=None, traj_len=None):
        # lag time in number of entries in assignment file (int).
        self.n_macro_states = n_macro_states
        self.assignments = None
        self.microstate_mapping_ = None
        self.MacroAssignments_ = None
        MarkovStateModel.__init__(self, lag_time=lag_time, n_macro_states=n_macro_states, traj_len=traj_len)
        self.n_neighbors = 5  # hard coding

    def fit(self, assignments):
        t0=time.time()
        self.assignments = assignments
        super(Ward, self).fit(assignments)
        self.run()
        t1=time.time()
        print "Ward Lumping running time:", t1-t0
        return self.MacroAssignments_

    def run(self):
        '''
        Do Ward Lumping.
        '''
        print "Doing Ward Lumping..."

        t0 = time.time()

        connectivity = kneighbors_graph(self.tProb_, n_neighbors=self.n_neighbors)
        # make connectivity symmetric
        #connectivity = 0.5 * (connectivity + connectivity.T)
        #print "Connectivity=", connectivity, "N_Neighbors=", self.n_fneighbors
        ward = cluster.AgglomerativeClustering(n_clusters=self.n_macro_states, linkage='ward', connectivity=connectivity)
        ward.fit(self.tProb_)
        t1 = time.time()
        print "Time:", t1-t0
        # print "Ward Results:"
        self.microstate_mapping_ = ward.labels_

        self.MacroAssignments_ = get_MacroAssignments(assignments=self.assignments,
                                                      microstate_mapping=self.microstate_mapping_)
        name = 'Ward_' + str(self.n_micro_states) + '_to_' + str(self.n_macro_states) + '_states_'
        OutputResult(homedir=self.homedir, tCount_=self.tCount_, tProb_=self.tProb_,
                     microstate_mapping_=self.microstate_mapping_, MacroAssignments_=self.MacroAssignments_, name=name)