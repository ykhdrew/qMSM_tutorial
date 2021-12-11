__author__ = 'stephen'
import numpy as np
import mdtraj as md
import os, sys


def get_subindices(assignments=None, state=None, samples=10):
    '''Get Subsamples assignments from same state'''
    assignments = np.array(assignments)
    if state is not None:
        indices = np.where(np.array(assignments) == state)[0]
    else:
        indices = range(0, len(assignments))
    if samples is not None:
        if len(indices) > samples:
            indices = np.random.choice(indices, size=samples)
    return indices

def output_trajs(trajs=None, assignments=None, n_states=None, samples=10, output_name='output', output_type='.pdb' ):
    for i in xrange(0, n_states):
        indices =get_subindices(assignments, i, samples)
        name = output_name + '_' + str(i) + '_' + output_type
        trajs[indices].save(name)

def split_assignments(assignments, lengths):
    '''Split a single long assignments into small segments by
    the lengths of original trajectories.'''
    if not sum(lengths) ==len(assignments):
        #raise Exception('The lengths are not equal!')
        raise Exception('sum(lengths)=%s, len(longlist)=%s' % (sum(lengths), len(assignments)))
    def find_position(x):
        length, cumlength = x
        return assignments[cumlength - length: cumlength]

    segments = [find_position(elem) for elem in zip(lengths, np.cumsum(lengths))]
    return segments

def merge_assignment_segments(segments, length):
    '''Merge small segments to a single long assignments.'''
    if not sum(length) == segments.size:
        raise Exception('The lengths are not equal!')
    assignments = []
    for i in segments:
        assignments.expand(i)
    return assignments
