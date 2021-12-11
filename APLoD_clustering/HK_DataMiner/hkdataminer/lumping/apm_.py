__author__ = 'stephen'
# ===============================================================================
# GLOBAL IMPORTS:
import os, sys
import time
import scipy.sparse.linalg
import numpy as np
# ===============================================================================
# LOCAL IMPORTS:
HK_DataMiner_Path = os.path.relpath(os.pardir)
sys.path.append(HK_DataMiner_Path)
from msm.msm import MarkovStateModel
from cluster import k_centers
from lumper_ import *
from lumping import PCCA
# ===============================================================================

class APM(MarkovStateModel):
    '''
    APM

    Parameters
    ----------
    n_macrostates : int
        The desired number of macrostates in the lumped model.
    lag_time: int
        The lag time of Markov State Model.
    homedir: str
        The dir of the results.
    traj_len:   {array-like}, shape=[n_samples]
            The length of each trajectory.
    '''
    def __init__(self, metric='rmsd', random_state=None, n_macro_states=6, max_iter=20, lag_time=1, sub_clus=2, homedir=None, traj_len=None, cut_by_mean=True):
        # lag time in number of entries in assignment file (int).
        self.cluster_centers_ = None
        self.labels_ = None
        self.n_macro_states = n_macro_states
        self.assignments = None
        self.microstate_mapping_ = None
        self.MacroAssignments_ = None
        self.cut_by_mean = cut_by_mean
        self.lag_time = lag_time
        self.sub_clus = sub_clus
        self.max_iter = max_iter
        self.micro_stack = []
        self.X=None
        self.traj_len =traj_len
        #MarkovStateModel.__init__(self, lag_time=lag_time, n_macro_states=n_macro_states, traj_len=traj_len)

    def fit(self, X, y=None):
        """Perform clustering.
        Parameters
        -----------
        X : array-like, shape=[n_samples, n_features]
            Samples to cluster.
        """
        #X = check_array(X)
        t0 = time.time()
        self.X=X
        self.run()
        print "APM clustering Time Cost:", t1 - t0
        return self

    def run(self):
        """Do the APM lumping.
        """
        print "Doing APM Clustering..."
        #Start looping for maxIter times
        n_macro_states = 1  #initialized as 1 because no macrostate exist in loop 0
        for iter in xrange(self.max_iter):
            #for k in xrange(n_macro_states):
                #TODO
            self.do_split()
                # set n_macro_states
                #n_macro_states = self.n_macro_states

            #do Lumping
            macro_lumper = PCCA(n_macro_states=self.n_macro_states, traj_len=self.traj_len, cut_by_mean=False)
            macro_lumper.fit(self.labels_)
            self.MacroAssignments_ = macro_lumper.MacroAssignments_
    
            print "Loop:", iter, "Metastability:", macro_lumper.metastability

        OutputResult(homedir=self.homedir, tCount_=self.tCount_, tProb_=self.tProb_,
                     microstate_mapping_=self.microstate_mapping_, MacroAssignments_=self.MacroAssignments_, name=name)

    def do_time_clustering(self):
        print "Stack:", self.micro_stack
        if not self.micro_stack:
            return
        else:
            micro_state = self.micro_stack[-1]  #last element of micro_stack
            if get_RelaxProb(micro_state, macro_state) > 0.6321206: #1-1/e
                #split if the relaxation time is too long
                self.do_split(micro_state=micro_state, sub_clus=self.sub_clus)
            else:
                #accept if the relaxation time is fine
                self.micro_stack.pop()

            self.do_time_clustering() #Note: recursion
        return

    def get_RelaxProb(self, micro_state=None, macro_state=None):
        X_len = len(self.X)
        count_trans = 0
        count_relax = 0
        for i in xrange(X_len):
            #if it starts at the desired state and ends at the same trajectory, count as one transition
            if self.labels_[i] == micro_state and self.MacroAssignments_[i] == macro_state:  #and self.labels_[i] ==self.labels_[i+self.lag_time]:
                count_trans += 1

                #if it does not end at the same state, count as one relaxation
                if self.labels_[i+self.lag_time] != micro_state or self.MacroAssignments_[i+self.lag_time] != macro_state:
                    count_relax += 1

            if count_trans > 0:
                return float(count_relax)/float(count_trans)
            else:
                return 1.0
            
    def do_split(self, micro_state = None, sub_clus=self.sub_clus):
        Micro_clusterer = KCenters(n_clusters=sub_clus, metric="rmsd", random_state=0)
        if micro_state is not None:
            sub_data_index = self.labels_.index(micro_state)
            sub_data = X[sub_data_index]

            #do split
            Micro_clusterer.fit(sub_data)
            sub_labels = Micro_clusterer.labels_

            #n_sub_states = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
            max_state = max(self.labels_)

            #add new cluster in to micro_stack
            for i in xrange(max_state+1, max_state+sub_clus):
                self.micro_stack.append(i)
            #rename the cluster number
            for i in xrange(len(sub_labels)):
                if sub_labels[i] is 0:
                    sub_labels[i] = micro_state
                else:
                    sub_data[i] += max_state
            #map new cluster back to micro assignment
            for i in xrange(len(sub_data)):
                self.labels_[ sub_data_index[i] ] = sub_labels[i]
        else:
           Micro_clusterer.fit(self.X)
           self.labels_ = Micro_clusterer.labels_





