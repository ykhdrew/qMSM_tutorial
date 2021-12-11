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
from lumper_ import *
# ===============================================================================

class PCCA(MarkovStateModel):
    '''
    Perron Cluster Cluster Analysis (PCCA) uses the eigenspectrum of a transition
    probability matrix to construct coarse-grained models, which states that a real
    square matrix with positive entries (e.g. a transition probability matrix) has
    a unique largest real eigenvalue and that the corresponding eigenvec- tor has
    strictly positive components. The term Perron Cluster refers to a set of
    eigenvalues clustered near the largest eigenvalue and separated from the rest
    of the eigenspectrum by a reasonable gap.

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
    def __init__(self, n_macro_states=None, lag_time=1, homedir=None, traj_len=None, cut_by_mean=True):
        # lag time in number of entries in assignment file (int).
        self.n_macro_states = n_macro_states
        self.assignments = None
        self.microstate_mapping_ = None
        self.MacroAssignments_ = None
        self.cut_by_mean = cut_by_mean
        MarkovStateModel.__init__(self, lag_time=lag_time, n_macro_states=n_macro_states, traj_len=traj_len)

    def fit(self, assignments):
        t0=time.time()
        self.assignments = assignments
        super(PCCA, self).fit(assignments)
        self.run()
        t1=time.time()
        print "PCCA Lumping running time:", t1-t0
        return self.MacroAssignments_

    def run(self, tolerance=1E-5):
        """Do the PCCA lumping.

        Notes
        -------
        1.  Iterate over the eigenvectors, starting with the slowest.
        2.  Calculate the spread of that eigenvector within each existing macrostate.
        3.  Pick the macrostate with the largest eigenvector spread.
        4.  Split the macrostate based on the sign of the eigenvector.
        """
        #Get Transition Count Matrix and Transition Probability Matrix
        #return eigenvalues and eigenvectors of TPM
        print "Doing PCCA Lumping..."
        #print "- getting Eigenvectors ...",
        t0 = time.time()
        self.eigenvalues, self.right_eigenvectors = self.get_righteigenvectors(self.tProb_, self.n_macro_states)
        print "right_eigenvectors_shape:", self.right_eigenvectors.shape
        print "right_eigenvectors type:", self.right_eigenvectors.dtype
        print self.right_eigenvectors[0:20, 0:3]
        print sum(self.right_eigenvectors[0,1:])
        #print "eigenvalues:"
        #print self.eigenvalues
        right_eigenvectors = self.right_eigenvectors[:, 1:]  # Extract non-perron eigenvectors
        print "right_eigenvectors_shape:", right_eigenvectors.shape

        t1 = time.time()
        #print "Time:", t1-t0
        #print "Done."
        microstate_mapping = np.zeros(self.n_micro_states, 'int32')

        # Function to calculate the spread of a single eigenvector.
        spread = lambda x: x.max() - x.min()

        print "PCCA Calculating..."
        #t0 = time.time()

        for i in range(self.n_macro_states - 1):
            if self.cut_by_mean is True:
                tolerance = np.mean(right_eigenvectors[:, i])
                print "mean:", tolerance, "max:", np.max(right_eigenvectors[:, i]), "min:", np.min(right_eigenvectors[:, i])
            v = right_eigenvectors[:, i]
            all_spreads = np.array([spread(v[microstate_mapping == k]) for k in range(i + 1)])
            state_to_split = np.argmax(all_spreads)
            microstate_mapping[(microstate_mapping == state_to_split) & (v >= tolerance)] = i + 1
        #t1 = time.time()
        #print "Time:", t1-t0

        #PCCA Results: microstate_mapping_
        self.microstate_mapping_ = microstate_mapping

        self.MacroAssignments_ = get_MacroAssignments(assignments=self.assignments,
                                                      microstate_mapping=self.microstate_mapping_)
        if self.cut_by_mean is True:
            name = 'PCCA_mean_' + str(self.n_micro_states) + '_to_' + str(self.n_macro_states) + '_states_'
        else:
            name = 'PCCA_' + str(self.n_micro_states) + '_to_' + str(self.n_macro_states) + '_states_'

        OutputResult(homedir=self.homedir, tCount_=self.tCount_, tProb_=self.tProb_,
                     microstate_mapping_=self.microstate_mapping_, MacroAssignments_=self.MacroAssignments_, name=name)


class PCCA_Plus(MarkovStateModel):
    '''
    Perron Cluster Cluster Analysis (PCCA) uses the eigenspectrum of a transition
    probability matrix to construct coarse-grained models, which states that a real
    square matrix with positive entries (e.g. a transition probability matrix) has
    a unique largest real eigenvalue and that the corresponding eigenvec- tor has
    strictly positive components. The term Perron Cluster refers to a set of
    eigenvalues clustered near the largest eigenvalue and separated from the rest
    of the eigenspectrum by a reasonable gap.

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
        super(PCCA, self).fit(assignments)
        self.run()
        t1=time.time()
        print "PCCA Lumping running time:", t1-t0
        return self.MacroAssignments_

    def run(self, tolerance=1E-5):
        """Do the PCCA lumping.

        Notes
        -------
        1.  Iterate over the eigenvectors, starting with the slowest.
        2.  Calculate the spread of that eigenvector within each existing macrostate.
        3.  Pick the macrostate with the largest eigenvector spread.
        4.  Split the macrostate based on the sign of the eigenvector.
        """
        #Get Transition Count Matrix and Transition Probability Matrix
        #return eigenvalues and eigenvectors of TPM
        print "Doing PCCA Lumping..."
        #print "- getting Eigenvectors ...",
        t0 = time.time()
        self.eigenvalues, self.right_eigenvectors = self.get_righteigenvectors(self.tProb_, self.n_macro_states)

        right_eigenvectors = self.right_eigenvectors[:, 1:]  # Extract non-perron eigenvectors
        print "right_eigenvectors_shape:", right_eigenvectors.shape
        t1 = time.time()
        #print "Time:", t1-t0
        #print "Done."
        microstate_mapping = np.zeros(self.n_micro_states, 'int32')

        # Function to calculate the spread of a single eigenvector.
        spread = lambda x: x.max() - x.min()

        print "PCCA Plus Calculating..."
        # t0 = time.time()

        for i in range(self.n_macro_states - 1):
            tolerance = np.mean(right_eigenvectors[:, i])
            print "mean:", tolerance, "max:", np.max(right_eigenvectors[:, i]), "min:", np.min(right_eigenvectors[:, i])
            v = right_eigenvectors[:, i]
            all_spreads = np.array([spread(v[microstate_mapping == k]) for k in range(i + 1)])
            state_to_split = np.argmax(all_spreads)
            microstate_mapping[(microstate_mapping == state_to_split) & (v >= tolerance)] = i + 1
        #t1 = time.time()
        #print "Time:", t1-t0

        #PCCA Results: microstate_mapping_
        self.microstate_mapping_ = microstate_mapping
        self.MacroAssignments_ = get_MacroAssignments(assignments=self.assignments,
                                                      microstate_mapping=self.microstate_mapping_)
        name = 'PCCAPlus_' + str(self.n_micro_states) + '_to_' + str(self.n_macro_states) + '_states_'
        OutputResult(homedir=self.homedir, tCount_=self.tCount_, tProb_=self.tProb_,
                     microstate_mapping_=self.microstate_mapping_, MacroAssignments_=self.MacroAssignments_, name=name)


class PCCA_Standard(MarkovStateModel):
    '''
    Perron Cluster Cluster Analysis (PCCA) uses the eigenspectrum of a transition
    probability matrix to construct coarse-grained models, which states that a real
    square matrix with positive entries (e.g. a transition probability matrix) has
    a unique largest real eigenvalue and that the corresponding eigenvec- tor has
    strictly positive components. The term Perron Cluster refers to a set of
    eigenvalues clustered near the largest eigenvalue and separated from the rest
    of the eigenspectrum by a reasonable gap.

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
    def __init__(self, n_macro_states=None, lag_time=1, homedir=None, traj_len=None):
        # lag time in number of entries in assignment file (int).
        self.n_macro_states = n_macro_states
        self.assignments = None
        self.microstate_mapping_ = None
        self.MacroAssignments_ = None
        MarkovStateModel.__init__(self, lag_time=lag_time, n_macro_states=n_macro_states, traj_len=traj_len)

    def getRightEigSolution(self, n):
        """Get largest eigenvalues and right eigenvectors of transition probability matrix reordered so eigenvalues are from largest to smallest (eigenvectors reordered to match eigenvalues).

        ARGUMENTS:
          n = number of eigenvalues/vectors to get (int)

        RETRUN: tuple where first part is eigenvalues and second part is eigenvectors. (tuple)
        """

        isSparse = scipy.sparse.issparse(self.tProb_)

        if n > self.n_macro_states:
          print "ERROR: There are only %d states and, thereofre, only %d eigenvalues.  Getting %d eigenvalues is impossible." % (self.n_macro_states, self.n_macro_states, n)
          sys.exit(1)
        if isSparse and n ==1:
          print "Shouldn't request a single eigenvalue/vector, there is a bug in scipy that will soon be fixed."
          sys.exit(1)

        eigSolution = None

        eigSolution = scipy.sparse.linalg.eigs(self.tProb_, n, which="LR")
        #if isSparse:
        #  eigSolution = scipy.sparse.linalg.eigs(self.tProb_.tocsr(), n, which="LR")
        #else:
        #  eigSolution = scipy.linalg.eig(self.tProb_)

        # reorder by eigenvalu (from largest to smallest) and only keep first n
        eigSolution = (eigSolution[0][(-eigSolution[0]).argsort()][0:n], eigSolution[1][:,(-eigSolution[0]).argsort()][:,0:n])
        return eigSolution

    def fit(self, assignments):
        t0=time.time()
        self.assignments = assignments
        super(PCCA_Standard, self).fit(assignments)
        self.run()
        t1=time.time()
        print "PCCA Standard Lumping running time:", t1-t0
        return self.MacroAssignments_

    def run(self, tolerance=1E-5):
        """Do the Standard PCCA lumping.

        Notes
        -------
        1.  Iterate over the eigenvectors, starting with the slowest.
        2.  Calculate the spread of that eigenvector within each existing macrostate.
        3.  Pick the macrostate with the largest eigenvector spread.
        4.  Split the macrostate based on the sign of the eigenvector.
        """
        #Get Transition Count Matrix and Transition Probability Matrix
        #return eigenvalues and eigenvectors of TPM
        print "Doing PCCA Standard Lumping..."
        #print "- getting Eigenvectors ...",
        t0 = time.time()
        print "Getting eigenvectors..."
        self.eigenvalues, self.right_eigenvectors = self.get_righteigenvectors(self.tProb_, self.n_macro_states)
        print "right_eigenvectors_shape:", self.right_eigenvectors.shape
        #self.right_eigenvectors = self.getRightEigSolution(self.n_macro_states)[1]
        #self.right_eigenvectors = self.right_eigenvectors[:, 1:]  # Extract non-perron eigenvectors
        #print "right_eigenvectors_shape:", self.right_eigenvectors.shape
        # by default eigenvectors column vectors, switch to row vectors for convenience
        right_eigenvectors = np.transpose(self.right_eigenvectors)

        print "right_eigenvectors_shape:", right_eigenvectors.shape

        t1 = time.time()
        print "Time:", t1-t0
        print "Done."
        microstate_mapping = np.zeros(self.n_micro_states, 'int32')

        # Function to calculate the spread of a single eigenvector.
        spread = lambda x: x.max() - x.min()

        print "PCCA Standard Calculating..."
        t0 = time.time()

        # build macrostates
        # start with one large macrostate and split self.n_macro_states-1=nEigPerron times
        for curNumMacro in range(1, self.n_macro_states):
            # find macrostate with largest spread in left eigenvector.
            # will split this one.
            # ignore first eigenvector since corresponds to equilibrium distribution
            maxSpread = -1         # max spread seen
            maxSpreadState = -1    # state with max spread
            for currState in range(curNumMacro):
                # find spread in components of eigenvector corresponding to current state
                myComponents = right_eigenvectors[curNumMacro][(microstate_mapping==currState).flatten()]
                maxComponent = max(myComponents)
                minComponent = min(myComponents)
                spread = maxComponent - minComponent

                # store if this is max spread seen so far
                if spread > maxSpread:
                    maxSpread = spread
                    maxSpreadState = currState

            # split the macrostate with the greatest spread.
            # microstates corresponding to components of macrostate eigenvector
            # greater than mean go in new
            # macrostate, rest stay in current macrostate
            meanComponent = np.mean(right_eigenvectors[curNumMacro][(microstate_mapping==maxSpreadState).flatten()]) #revised by Wei, June 23, night
            print "mean:", meanComponent, "max:", maxComponent, "min:", minComponent
            #meanComponent = 1E-5
            newMacrostateIndices = (right_eigenvectors[curNumMacro] >= meanComponent)*(microstate_mapping==maxSpreadState)
            microstate_mapping[newMacrostateIndices] = curNumMacro
        t1 = time.time()
        print "Time:", t1-t0

        #PCCA Results: microstate_mapping_
        self.microstate_mapping_ = microstate_mapping

        self.MacroAssignments_ = get_MacroAssignments(assignments=self.assignments,
                                                      microstate_mapping=self.microstate_mapping_)
        name = 'PCCAStandard_' + str(self.n_micro_states) + '_to_' + str(self.n_macro_states) + '_states_'
        OutputResult(homedir=self.homedir, tCount_=self.tCount_, tProb_=self.tProb_,
                     microstate_mapping_=self.microstate_mapping_, MacroAssignments_=self.MacroAssignments_, name=name)

        self.metastability = self.tProb_.diagonal().sum()
        self.metastability /= len(self.tProb_)

