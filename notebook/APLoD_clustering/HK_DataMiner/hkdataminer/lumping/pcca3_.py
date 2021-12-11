__author__ = 'stephen'
###############################################################################
#      Filename:  lumper.py
#       Created:  2015-04-24 16:48
#        Author:  Tiago Lobato Gimenes        (tlgimenes@gmail.com)
###############################################################################

###############################################################################

"""
Classes for lumping the microstates assignments into macrostates assignments
"""

###############################################################################
# Global imports
import msmbuilder.lumping
import numpy as np

###############################################################################

class PCCA3:
    """
    Constructs lumper with n macrostates
    """
    def __init__(self, n_macrostates):
        self._n_macrostates = n_macrostates

        self._lmpr = msmbuilder.lumping.PCCA(self._n_macrostates)
        #self._lmpr = msmbuilder.lumping.PCCAPlus(self._n_macrostates)

    """
    Fits the microstates vector into the model with outliers given by outliers
    """
    """
    def fit(self, microstates, traj_size, outliers=-1):
        self._n_traj = len(microstates) / traj_size

        micro = list()
        for i in range(self._n_traj):
            traj = np.array(microstates[i*traj_size:(i+1)*traj_size], dtype='int');
            micro.append(traj[np.where(traj != outliers)]);
            #micro.append(traj);

        self._lmpr.fit(micro);
    """
    def fit(self, microstates, traj_size, outliers=-1):
        self._n_traj = len(microstates) / traj_size

        micro = list();
        for i in range(self._n_traj):
            traj = np.array(microstates[i*traj_size:(i+1)*traj_size], dtype='int')
            beg = 0
            end = len(traj)
            for i in np.where(traj == -1)[0]:
                if i != 0:
                    micro.append(traj[beg:i])
                    beg = i+1
            micro.append(traj[beg:end])
        self._lmpr.fit(micro)
    """def fit(self, microstates, traj_size, outliers=-1):
        self._n_traj = len(microstates) / traj_size

        min_size = 0;
        for i in range(self._n_traj):
            traj = np.array(microstates[i*traj_size:(i+1)*traj_size], dtype='int');
            traj = traj[np.where(traj == outliers)];
            if(-len(traj) < min_size):
                min_size = -len(traj);
        min_size += traj_size;

        micro = list();
        for i in range(self._n_traj):
            traj = np.array(microstates[i*traj_size:(i+1)*traj_size], dtype='int');
            #micro.append(traj);
            traj = traj[np.where(traj != outliers)]
            micro.append(traj[0:min_size]);

        self._lmpr.fit(micro);
    """

    """
    Transform microstates to macrostates leaving outliers the way they are
    """
    def transform(self, microstates, outliers=-1):
        self._micro_to_macro = dict((key, self._lmpr.microstate_mapping_[val])
                                    for (key, val) in self._lmpr.mapping_.items())

        macrostates = np.ndarray((len(microstates),), dtype='int')
        for i in range(len(microstates)):
            if microstates[i] != outliers and microstates[i] in self._micro_to_macro:
                macrostates[i] = self._micro_to_macro[microstates[i]]
            else:
                macrostates[i] = outliers

        return macrostates



class PCCA_Plus:
    """
    Constructs lumper with n macrostates
    """
    def __init__(self, n_macrostates):
        self._n_macrostates = n_macrostates

        #self._lmpr = msmbuilder.lumping.PCCA(self._n_macrostates)
        self._lmpr = msmbuilder.lumping.PCCAPlus(self._n_macrostates)

    """
    Fits the microstates vector into the model with outliers given by outliers
    """
    """
    def fit(self, microstates, traj_size, outliers=-1):
        self._n_traj = len(microstates) / traj_size

        micro = list()
        for i in range(self._n_traj):
            traj = np.array(microstates[i*traj_size:(i+1)*traj_size], dtype='int');
            micro.append(traj[np.where(traj != outliers)]);
            #micro.append(traj);

        self._lmpr.fit(micro);
    """
    def fit(self, microstates, traj_size, outliers=-1):
        self._n_traj = len(microstates) / traj_size;

        micro = list();
        for i in range(self._n_traj):
            traj = np.array(microstates[i*traj_size:(i+1)*traj_size], dtype='int');
            beg = 0;
            end = len(traj)
            for i in np.where(traj == -1)[0]:
                if i != 0:
                    micro.append(traj[beg:i]);
                    beg = i+1;
            micro.append(traj[beg:end]);
        self._lmpr.fit(micro);
    """def fit(self, microstates, traj_size, outliers=-1):
        self._n_traj = len(microstates) / traj_size

        min_size = 0;
        for i in range(self._n_traj):
            traj = np.array(microstates[i*traj_size:(i+1)*traj_size], dtype='int');
            traj = traj[np.where(traj == outliers)];
            if(-len(traj) < min_size):
                min_size = -len(traj);
        min_size += traj_size;

        micro = list();
        for i in range(self._n_traj):
            traj = np.array(microstates[i*traj_size:(i+1)*traj_size], dtype='int');
            #micro.append(traj);
            traj = traj[np.where(traj != outliers)]
            micro.append(traj[0:min_size]);

        self._lmpr.fit(micro);
    """

    """
    Transform microstates to macrostates leaving outliers the way they are
    """
    def transform(self, microstates, outliers=-1):
        self._micro_to_macro = dict((key, self._lmpr.microstate_mapping_[val])
                                    for (key, val) in self._lmpr.mapping_.items())

        macrostates = np.ndarray((len(microstates),), dtype='int');
        for i in range(len(microstates)):
            if microstates[i] != outliers and microstates[i] in self._micro_to_macro:
                macrostates[i] = self._micro_to_macro[microstates[i]];
            else:
                macrostates[i] = outliers;

        return macrostates;

###############################################################################

# -*- coding: utf-8 -*-
# vim:fenc=utf-8

###############################################################################