__author__ = 'stephen'
import scipy.io
import scipy.sparse
import numpy as np
from msm import MarkovStateModel
from utils import plot_matrix
def get_MacroAssignments(assignments=None, microstate_mapping=None, outlier=-1):
    '''

    :param assignments: The micro-states assignments
    :param microstate_mapping: The result after lumping
    :param outlier: The outlier, default is -1
    :return: MacroAssignments: The result after map microstates onto macrostates
    '''

    MacroAssignments = []
    for i in range(len(assignments)):
        if assignments[i] != outlier:
            microCenter = assignments[i]
            macroCenter = microstate_mapping[ microCenter ]
            MacroAssignments.append(macroCenter)
        else:
            MacroAssignments.append(outlier)
    return np.array(MacroAssignments)

def OutputResult(homedir='.', tCount_=None, tProb_=None, microstate_mapping_=None, MacroAssignments_=None, name=None):
    '''
    :param homedir: The dir of the results.
    :param tCount_: Transition Count Matrix
    :param tProb_:  Transition Probability Matrix
    :param microstate_mapping_: The result after lumping
    :param MacroAssignments_:  The result after map microstates onto macrostates
    :return:
    '''
    tCountDir       = homedir + "/" + name + "tCounts.mtx"
    tProbDir        = homedir + "/" + name + "tProb.mtx"
    MappingDir      = homedir + "/" + name + "microstate_mapping.txt"
    MacroAssignDir  = homedir + "/" + name + "MacroAssignments.txt"
    if tCount_ is not None:
        scipy.io.mmwrite(tCountDir, scipy.sparse.csr_matrix(tCount_), field='integer')
    if tProb_ is not None:
        scipy.io.mmwrite(tProbDir, scipy.sparse.csr_matrix(tProb_))
    if microstate_mapping_ is not None:
        np.savetxt(MappingDir, microstate_mapping_, fmt="%d")
    if MacroAssignments_ is not None:
        np.savetxt(MacroAssignDir, MacroAssignments_, fmt="%d")
    #Evaluate lumping results
    Evaluate_Result(homedir='.', tProb_=tProb_, lag_time=1, microstate_mapping_=microstate_mapping_, MacroAssignments_=MacroAssignments_, name=name)


def Evaluate_Result(homedir='.', tProb_=None, lag_time=1, microstate_mapping_=None, MacroAssignments_=None, name=None):

    MacroMSM = MarkovStateModel(lag_time=lag_time)
    MacroMSM.fit(MacroAssignments_)
    Macro_tProb_ = MacroMSM.tProb_
    Macro_tCount_ = MacroMSM.tCount_
    #plot_matrix(labels=None, tProb_=tProb_, name=name)

    #Calculate metastablilty
    print "Calculating Metastablilty and Modularity..."
    metastability = Macro_tProb_.diagonal().sum()
    metastability /= len(Macro_tProb_)
    print "Metastability:", metastability

    #Calculate modularity
    micro_tProb_ = tProb_
    degree = micro_tProb_.sum(axis=1) #row sum of tProb_ matrix
    total_degree = degree.sum()

    modularity = 0.0
    len_mapping = len(microstate_mapping_)
    for i in xrange(len_mapping):
        state_i = microstate_mapping_[i]
        for j in xrange(len_mapping):
            state_j = microstate_mapping_[j]
            if state_i == state_j:
                modularity += micro_tProb_[i, j] - degree[i]*degree[j]/total_degree
    modularity /= total_degree
    print "Modularity:", modularity

    Macro_tCountDir       = homedir + "/" + name + "Macro_tCounts.mtx"
    Macro_tProbDir        = homedir + "/" + name + "Macro_tProb.mtx"
    Metastability_Modularity_Dir = homedir + "/" + name + "Metastability_Modularity.txt"

    np.savetxt(Metastability_Modularity_Dir, [metastability, modularity], fmt="%lf")
    scipy.io.mmwrite(Macro_tCountDir, scipy.sparse.csr_matrix(Macro_tCount_), field='integer')
    scipy.io.mmwrite(Macro_tProbDir, scipy.sparse.csr_matrix(Macro_tProb_))
    #Plot tProb matrix
    plot_matrix(tProb_=Macro_tProb_, name=name)
