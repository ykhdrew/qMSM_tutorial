__author__ = 'stephen'
# ===============================================================================
# GLOBAL IMPORTS:
import os,sys
import numpy as np
import argparse

# ===============================================================================
# LOCAL IMPORTS:
HK_DataMiner_Path = os.path.relpath(os.pardir)
#HK_DataMiner_Path = os.path.abspath("/home/stephen/Dropbox/projects/work-2015.5/HK_DataMiner/")
print HK_DataMiner_Path
sys.path.append(HK_DataMiner_Path)
from cluster import DBSCAN
from utils import XTCReader, plot_cluster
# ===============================================================================

outliers = -1

def merge_assignments(new_assignments, old_assignments):
    # Number of clusters in assignments, ignoring noise if present.
    #clusters_size = len(set(old_assignments)) - (1 if -1 in old_assignments else 0)
    clusters_size = np.max(old_assignments) + 1
    max_clust_id = clusters_size
    print "max_clust_id:", max_clust_id
    count_first = [0] * clusters_size
    count_second = [0] * clusters_size

    old_assignments_size = len(old_assignments)
    # new_assignments_size = len(new_assignments)
    for i in xrange(0, old_assignments_size):
        if old_assignments[i] != outliers:
            if new_assignments[i] != outliers:
                count_first[old_assignments[i]] += 1
            count_second[old_assignments[i]] += 1

    # Percentage
    percentage = [0.0] * clusters_size
    for i in xrange(0, clusters_size):
        if count_second[i] is 0:
            percentage[i] = 0.0
        else:
            percentage[i] = float(count_first[i])/float(count_second[i])

    # Starting assignment
    for i in xrange(0, old_assignments_size):
        if old_assignments[i] != outliers and percentage[old_assignments[i]] > 0.7:
            if new_assignments[i] != outliers:
                old_assignments[i] = new_assignments[i] + max_clust_id
                # print old_assignments[i]
            # else:  #if wang to remove outliers in the iterations
            #    old_assignments[i] = outliers

    return old_assignments

def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('-t',   '--trajListFns', default = 'trajlist',
                     help='List of trajectory files to read in, separated by spaces.')
    cli.add_argument('-a', '--atomListFns', default='atom_indices',
                     help='List of atom index files to read in, separated by spaces.')
    cli.add_argument('-g',   '--topology', default='native.pdb', help='topology file.')
    cli.add_argument('-o',   '--homedir',  help='Home dir.', default=".", type=str)
    cli.add_argument('-e',   '--iext', help='''The file extension of input trajectory
                     files.  Must be a filetype that mdtraj.load() can recognize.''',
                     default="xtc", type=str)
    cli.add_argument('-n',   '--n_clusters', help='''n_clusters.''',
                     default=100, type=int)
    cli.add_argument('-m',   '--n_macro_states', help='''n_macro_states.''',
                     default=6, type=int)
    cli.add_argument('-s',   '--stride', help='stride.',
                     default=None, type=int)

    args = cli.parse_args()
    trajlistname = args.trajListFns
    atom_indicesname = args.atomListFns
    trajext = args.iext
    File_TOP = args.topology
    homedir = args.homedir
    n_clusters = args.n_clusters
    n_macro_states = args.n_macro_states
    stride = args.stride
    # ===========================================================================
    # Reading Trajs from XTC files
    #print "stride:", stride
    #trajreader = XTCReader(trajlistname, atom_indicesname, homedir, trajext, File_TOP, nSubSample=stride)
    #trajs = trajreader.trajs
    #print trajs
    #traj_len = trajreader.traj_len
    #np.savetxt("./traj_len.txt", traj_len, fmt="%d")

    if os.path.isfile("./phi_angles.txt") and os.path.isfile("./psi_angles.txt") is True:
        phi_angles = np.loadtxt("./phi_angles.txt", dtype=np.float32)
        psi_angles = np.loadtxt("./psi_angles.txt", dtype=np.float32)
    else:
        phi_angles, psi_angles = trajreader.get_phipsi(trajs, psi=[6, 8, 14, 16], phi=[4, 6, 8, 14])
        #phi_angles, psi_angles = trajreader.get_phipsi(trajs, psi=[5, 7, 13, 15], phi=[3, 5, 7, 13])
        np.savetxt("./phi_angles.txt", phi_angles, fmt="%f")
        np.savetxt("./psi_angles.txt", psi_angles, fmt="%f")

    phi_psi=np.column_stack((phi_angles, psi_angles))

    n_samples = 1000
    percent = 0.9
    import random
    whole_samples = random.sample(phi_psi, n_samples)
    #print whole_samples
    from metrics.pairwise import pairwise_distances
    sample_dist_metric = pairwise_distances(whole_samples, whole_samples, metric='rmsd')
    print sample_dist_metric.shape
    sample_dist = []
    for i in xrange(0, n_samples):
        for j in xrange(i+1, n_samples):
            sample_dist.append(sample_dist_metric[i, j])
    sorted_sample_dist=np.sort(sample_dist)
    print "Len of samples:", len(sorted_sample_dist), np.max(sorted_sample_dist), np.min(sorted_sample_dist)


    eps_list = []
    len_samples = len(sorted_sample_dist)
    for percent in [0.40, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.025, 0.010, 0.008, 0.005, 0.003,
                    0.001, 0.0005, 0.0003, 0.0001, 0.00005, 0.00001]:
        percent /= 10.0
        index = int(round(len_samples*percent))
        if index == len_samples:
            index -= 1
        dc = sorted_sample_dist[index]
        #print index, sorted_sample_dist[index]
        eps_list.append(dc)
    print eps_list

    # from sklearn.neighbors import NearestNeighbors
    # print len(phi_psi)
    # neighborhoods_model = NearestNeighbors(n_neighbors=len(phi_psi), algorithm='kd_tree')
    # neighborhoods_model.fit(phi_psi)
    # #distances, indices = neighborhoods_model.kneighbors(phi_psi)
    # distances, indices = neighborhoods_model.kneighbors(phi_psi, 5)
    # print distances



    #print phi_psi
    # ===========================================================================
    # do Clustering using MR -DBSCAN method
    clustering_name = "mr-dbscan_iter_"
    potential = True
    # potential = False
    eps = eps_list[0]
    min_samples = 5
    print "Running first calculation"
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='rmsd').fit(phi_psi)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    old_assignments = db.labels_
    n_microstates = len(set(old_assignments)) - (1 if -1 in old_assignments else 0)
    print('Estimated number of clusters: %d' % n_microstates)
    iter_name = clustering_name + '0' + '_eps_' + str(eps) + '_min_samples_' + str(min_samples) + '_n_states_' + str(n_microstates)
    plot_cluster(labels=old_assignments, phi_angles=phi_angles, psi_angles=psi_angles, name=iter_name, potential=potential)

    n_iterations = len(eps_list)
    print "n_iterations:", n_iterations
    #eps_list = [3.0, 2.0, 1.0, 0.8, 0.5]
    min_samples_list = [3, 3, 3, 3, 3, 2, 2]

    for i in xrange(1, n_iterations):
        eps = eps_list[i]
        #min_samples = min_samples_list[i]
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='rmsd').fit(phi_psi)
        print "Iter:", i, "Running MR-DBSCAN at eps:", eps, 'min_sampes:', min_samples
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        new_assignments = db.labels_
        old_assignments = merge_assignments(new_assignments, old_assignments)
        n_microstates = len(set(old_assignments)) - (1 if -1 in old_assignments else 0)
        print('Estimated number of clusters: %d' % n_microstates)
        iter_name = clustering_name + str(i) + '_eps_' + str(eps) + '_min_samples_' + str(min_samples) + '_n_states_' + str(n_microstates)
        plot_cluster(labels=old_assignments, phi_angles=phi_angles, psi_angles=psi_angles, name=iter_name, potential=potential)
    labels = old_assignments
    print labels
    n_microstates = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_microstates)

    #cluster_centers_ = cluster.cluster_centers_
    # plot micro states
    clustering_name = "mr-dbscan_n_" + str(n_microstates)
    np.savetxt("assignments_"+clustering_name+".txt", labels, fmt="%d")
    #np.savetxt("cluster_centers_"+clustering_name+".txt", cluster_centers_, fmt="%d")

    plot_cluster(labels=labels, phi_angles=phi_angles, psi_angles=psi_angles, name=clustering_name, potential=potential)

if __name__ == "__main__":
    main()