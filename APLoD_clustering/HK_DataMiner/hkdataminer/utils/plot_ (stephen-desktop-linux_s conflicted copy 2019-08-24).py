__author__ = 'stephen'
import numpy as np
import scipy.io
import scipy.sparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.pylab as pylab
from .utils import get_subindices
import matplotlib.ticker as mtick
from collections import Counter
from sklearn.neighbors.kde import KernelDensity
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
def plot_cluster(labels, phi_angles, psi_angles, name, outliers=-1, step=1, potential=False):
    '''
    :param labels: the assignments after clustering or lumping
    :param phi_angles: the phi angles
    :param psi_angles: the psi angles
    :param name: the name of the result pictures
    :param outliers: outliers default is -1
    :return: None
    '''

    clusters = np.unique(labels)
    plt.rc("font", size=10)
    if step > 1:
        clusters = clusters[0:len(clusters):step]
    colors_jet = plt.cm.jet(np.linspace(0, 1, np.max(clusters)+1))
    if potential is False: #plot Alanine Dipeptide
        for i in clusters:
            if i != outliers:
                point = np.where(labels == i)
                plt.plot(phi_angles[point], psi_angles[point], '.', markersize=1.0, alpha=0.7)#, color=colors_jet[i])
            #else:
            #    point = np.where(labels == i)
            #    plt.plot(phi_angles[point], psi_angles[point], '.', markersize=1.0, alpha=0.7, color='black')  # , color=colors_jet[i])
        plt.title("Alanine Dipeptide " + name + " states", fontsize=10)
        plt.xlim([-180, 180])
        plt.ylim([-180, 180])
        plt.xticks([-110, -60, 0, 60, 120])
        plt.yticks([-120, -60, 0, 60, 120])
    else:  # if plot 2D potential
        plt.figure(figsize=(10, 10))
        for i in clusters:
            if i != outliers:
                plt.plot(phi_angles[np.where(labels == i)],
                        psi_angles[np.where(labels == i)], '.', markersize=1.0, alpha=0.7) #markersize=20.0, color=colors_jet[i])
                #plt.plot(phi_angles[np.where(labels == i)],
                #        psi_angles[np.where(labels == i)],
                #        '.', color=colors_jet[i], label='State %d' % i)
        #plt.title("2D potential " + name + " states", fontsize=20)
        plt.xlim([-75, 75])
        plt.ylim([-75, 75])
        plt.xticks([-50, 0, 50])
        plt.yticks([-50, 0, 50])


    plt.xlabel(r"$\phi$", fontsize=25)
    plt.ylabel(r"$\psi$", fontsize=25)
    # Save the result figure
    plt.savefig('./'+name+'.png', dpi=400)
    plt.close()
    #plt.show()


def plot_each_cluster(labels, phi_angles, psi_angles, name, outliers=-1, step=1):
    '''
    :param labels: the assignments after clustering or lumping
    :param phi_angles: the phi angles
    :param psi_angles: the psi angles
    :param name: the name of the result pictures
    :param outliers: outliers default is -1
    :return: None
    '''

    clusters = np.unique(labels)
    if step > 1:
        clusters = clusters[0:len(clusters):step]
    colors_jet = plt.cm.jet(np.linspace(0, 1, np.max(clusters)+1))

    for i in np.unique(clusters):
        if i != outliers:
            plt.plot(phi_angles[np.where(labels == i)],
                    psi_angles[np.where(labels == i)],
                    'x', color=colors_jet[i], label='State %d' % i)
            #plt.title("Alanine Dipeptide " + name + " state_" + str(i))
            plt.xlabel(r"$\phi$")
            plt.ylabel(r"$\psi$")

            plt.xlim([-180, 180])
            plt.ylim([-180, 180])
            plt.xticks([-120, -60, 0, 60, 120])
            plt.yticks([-120, -60, 0, 60, 120])
            # Save the result figure
            plt.savefig('./'+ name + " state_" + str(i)+'.png', dpi = 400)
            plt.close()
            #plt.show()


def contour_cluster(labels, phi_angles, psi_angles, name, outliers=-1):
    '''
    :param labels: the assignments after clustering or lumping
    :param phi_angles: the phi angles
    :param psi_angles: the psi angles
    :param name: the name of the result pictures
    :param outliers: outliers default is -1
    :return: None
    '''

    # lables_array = np.array(labels)
    # colors_jet = plt.cm.jet(np.linspace(0, 1, np.max(lables_array)+1))

    for i in np.unique(labels):
        #if i != outliers:
        if i == 1:
            print("i=", i)
            x = phi_angles[np.where(labels == i)]
            y = psi_angles[np.where(labels == i)]
            indices = get_subindices(assignments=x, state=None, samples=1000)
            x = x[indices]
            y = y[indices]
            X, Y= np.meshgrid(x, y)
            positions = np.vstack([X.ravel(), Y.ravel()])
            values = np.vstack([x, y])
            kernel = stats.gaussian_kde(values)
            Z = np.reshape(kernel(positions).T, X.shape)
            #kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
            #kde_results = kde.score_samples([x,y])
            #X, Y, Z = np.meshgrid(x, y, kde_results)
            #Z = np.reshape(kernel([x,y]).T, x.shape)
            #Z1 = mlab.bivariate_normal(X, Y, 5.0, 5.0, 0.0, 0.0)
            #Z2 = mlab.bivariate_normal(X, Y, 7.5, 2.5, 5, 5)
            # difference of Gaussians
            #Z = 10.0 * (Z2 - Z1)
            #step = Z.max()-Z.min()/10
            #print "Z min:",Z.min(), "Z.max:", Z.max(), "step:", step

            #levels = np.arange(Z.min(),  Z.min(), Z.max())
            #print levels
            plt.contour(X, Y, Z, origin='lower') #, linewidths=Z.min(), levels=levels)

    plt.title("Alanine Dipeptide " + name + " states")
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$\psi$")

    plt.xlim([-180, 180])
    plt.ylim([-180, 180])
    # Save the result figure
    plt.savefig('./'+name+'.png', dpi=400)
    plt.close()
    #plt.show()

def plot_matrix(tProb_=None, name=None):
    '''
    if labels is not None:
        n_states = len(set(labels)) - (1 if -1 in labels else 0)
        print 'n_states=', n_states
        #diagC = tProb_.diagonal()
        length = len(labels)
        print "length=", length
        Cmn = scipy.sparse.lil_matrix(n_states, n_states, dtype=np.float32)
        Cmn = np.zeros((n_states, n_states))
        print "size of tProb", tProb_.shape
        if scipy.sparse.issparse(tProb_):
            tProb_ = tProb_.todense()
        for i in xrange(length):
            for j in xrange(length):
                Cmn[labels[i], labels[j]] += tProb_[i, j]

        #for i in xrange(n_states):
            #Cmn[i,i] += diagC[i]
        #    for j in xrange(n_states):
        #        Cmn[i, j] += Cmn[j, i]
        #        Cmn[j, i] = Cmn[i, j]

        for j in xrange(n_states):
            sum_row = np.sum(Cmn[j,:])
            if sum_row is not 0:
                Cmn[j,:] /= sum_row

        pylab.matshow(Cmn, cmap=plt.cm.OrRd)
    else:
    '''
    pylab.matshow(tProb_, cmap=plt.cm.OrRd)
    plt.colorbar()
    #pylab.show()
    plt.savefig('./' + name + 'Matrix.png', dpi=400)
    plt.close()

def plot_block_matrix(labels, tProb_, name='BlockMatrix'):
    print("Plot Block Matrix")
    indices = np.argsort(labels)
    #print indices
    block_matrix = tProb_[:,indices]
    block_matrix = block_matrix[indices,:]
    block_matrix = 1 - block_matrix
    #print block_matrix
    pylab.matshow(block_matrix, cmap=plt.cm.OrRd)
    plt.colorbar()
    plt.savefig('./' + name + '.png', dpi=400)
    #pylab.show()
    plt.close()

def plot_cluster_size_distribution(populations, name='Populations'):
    fig = plt.figure(1, (10,6))
    distrib = fig.add_subplot(1,1,1)
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    distrib.yaxis.set_major_formatter(xticks)
    plt.rc("font", size=30)
    plt.title('Cluster size distributions', fontsize=20)
    distrib.grid(True)
    X = range(len(populations))
    X_xtick = ['']
    for i in xrange(1, len(populations)+1):
        xx = '$10^' + str(i) + '$'
        X_xtick.append(xx)
    print(X_xtick)
    #plt.xticks(X , ('$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'))
    plt.xticks(np.arange(len(populations)+1), X_xtick)
    plt.ylabel(r"Probability")
    plt.ylim([0,100])
    print("X:", X)
    distrib.bar(X, populations*100, facecolor='black', edgecolor='white', width=1.0) #facecolor='#f78181',
    plt.savefig('./' + name + '_Distribution.png', dpi=400)
    plt.close()
    #plt.show()


def plot_compare_cluster_size_distribution(populations_1, populations_2, name='Populations'):
    fig = plt.figure(1, (10,8))
    distrib = fig.add_subplot(1,1,1)
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    distrib.yaxis.set_major_formatter(xticks)
    bar_width = 0.45
    plt.rc("font", size=20)
    #plt.title('Cluster size distributions', fontsize=20)
    distrib.grid(True)
    X = np.arange(len(populations_1))
    X_xtick = ['']
    for i in xrange(1, len(populations_1)+1):
        xx = '$10^' + str(i) + '$'
        X_xtick.append(xx)
    print(X_xtick)
    #plt.xticks(X , ('$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'))

    print("X:", X)
    distrib.bar(X, populations_1*100, facecolor='black', edgecolor='white', width=bar_width,label="kNN Density Peaks 3645 states") #facecolor='#f78181',

    #  populations_2
    #X = range(len(populations_2))
    X_xtick = ['']
    for i in xrange(1, len(populations_2)+1):
        xx = '$10^' + str(i) + '$'
        X_xtick.append(xx)
    print(X_xtick)
    #plt.xticks(X , ('$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'))

    print("X:", X)
    distrib.bar(X+bar_width, populations_2*100, facecolor='gray', edgecolor='white', width=bar_width, label="kNN Density Peaks 117 states") #facecolor='#f78181',
    plt.xticks(np.arange(len(populations_1)+1+bar_width), X_xtick)

    #plt.ylabel(r"Fraction number of clusters")
    plt.ylabel(r"Probability")
    plt.ylim([0,60])
    plt.legend()
    plt.savefig('./' + name + '_Distribution.png', dpi=400)
    plt.close()
    #plt.show()

#From Wang Wei's code
def plot_landscape(labels=None, phi_angles=None, psi_angles=None, phi_ctr=None, psi_ctr=None, name='Energy_Landscape', bins=80, potential=False):
    H, xedges, yedges = np.histogram2d(psi_angles, phi_angles, bins=bins)
    #since we calculate total number in 10 interval, thus bin of every dimension must be 36
    #If element in H is zero, set the final energy to be 9
    plt.rc("font", size=25)
    maxH = np.max(H)
    for i in range(len(H)):
        for j in range(len(H)):
            if H[i][j]==0:
                H[i][j]=9
            else:
                H[i][j] = -np.log(H[i][j]/maxH)

    #H = -np.log(H/np.max(H))
    extent =[np.min(xedges), np.max(xedges), np.min(yedges), np.max(yedges)]
    plt.figure(figsize=(12, 12))
    plt.imshow(H, extent=extent, origin="lower", cmap=plt.cm.gray)  #plt.cm.jet
    #plot cluster centers on landscape
    if labels is not None:
        plt.plot(phi_ctr, psi_ctr, '.', markersize=10, color='r')

    distribution = np.array([0,0,0,0,0,0,0,0,0,0], dtype=np.float64)
    #print "len phi_ctr", len(phi_ctr)
    #print "shape of xedges", xedges.shape
    for i in range(0, len(phi_angles)):
        if psi_angles[i] > 179.0:
            index_x = np.where(xedges > 179.0)[0][0] - 1
        else:
            index_x = np.where(xedges > psi_angles[i])[0][0] - 1
        if phi_angles[i] > 179.0:
            index_y = np.where(yedges > 179.0)[0][0] - 1
        else:
            index_y = np.where(yedges > phi_angles[i])[0][0] - 1

        index_distrib = int(H[index_x][index_y])
        distribution[index_distrib] += 1
    distribution /= len(phi_angles)
    print(distribution)
       # print "clenter:", i, "[", phi_ctr,",", psi_ctr,"]", "H=", H[index_x][index_y]
    plt.xlabel('$\phi$', fontsize=20)
    plt.ylabel('$\Psi$', fontsize=20)

    cbar = plt.colorbar(shrink=0.77)
    #plt.title('Free energy landscape', fontsize=20)
    cbar.set_label("$k_B T$", size=20)
    cbar.ax.tick_params(labelsize=20)
    if potential is False:
        plt.xlim([-180, 180])
        plt.ylim([-180, 180])
        plt.xticks([-120, -60, 0, 60, 120])
        plt.yticks([-120, -60, 0, 60, 120])
    else:
        plt.xlim([-75, 75])
        plt.ylim([-75, 75])
        plt.xticks([-50, 0, 50])
        plt.yticks([-50, 0, 50])
    plt.savefig('./' + name + '.png', dpi=400)
    #plt.show()
    plt.close()

    #Cluster Centers on Free energy landscape distribution
    fig = plt.figure(1, (10,6))
    plt.rc("font", size=15)
    distrib = fig.add_subplot(1,1,1)
    distrib.grid(True)
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    distrib.yaxis.set_major_formatter(xticks)
    plt.title('Cluster Centers on Free energy landscape distribution', fontsize=20)
    plt.xlabel("$k_B T$")
    plt.ylabel(r"Probability")
    plt.ylim([0, 100])
    plt.xticks(np.arange(11), ('', '1', '', '3', '', '5', '', '7', '', '9', ''))
    distrib.bar(np.arange(10), distribution*100, facecolor='black', edgecolor='white', width=1.0) #facecolor='#f78181'
    plt.savefig('./' + name + '_Distribution.png', dpi=400)
    #plt.show()
    plt.close()

def plot_compare_distribution(labels_1=None, labels_2=None, phi_angles=None, psi_angles=None, phi_ctr_1=None, psi_ctr_1=None, phi_ctr_2=None, psi_ctr_2=None, name='Energy_Landscape', bins=36, potential=False):
    H, xedges, yedges = np.histogram2d(psi_angles, phi_angles, bins=bins)
    #since we calculate total number in 10 interval, thus bin of every dimension must be 36
    #If element in H is zero, set the final energy to be 9
    plt.rc("font", size=25)
    maxH = np.max(H)
    for i in range(len(H)):
        for j in range(len(H)):
            if H[i][j]==0:
                H[i][j]=9
            else:
                H[i][j] = -np.log(H[i][j]/maxH)

    #H = -np.log(H/np.max(H))
    #extent =[np.min(xedges), np.max(xedges), np.min(yedges), np.max(yedges)]
    #plt.figure(figsize=(10, 10))
    #plt.imshow(H, extent=extent, origin="lower", cmap=plt.cm.gray)  #plt.cm.jet
    #plot cluster centers on landscape
    #if labels_1 is not None:
    #    plt.plot(phi_ctr_1, psi_ctr_1, '*', markersize=8, color='r')

    distribution_1 = np.array([0,0,0,0,0,0,0,0,0,0], dtype=np.float64)
    for i in xrange(0, len(phi_ctr_1)):
        if psi_ctr_1[i] > 179.0:
            index_x = np.where(xedges > 179.0)[0][0] - 1
        else:
            index_x = np.where(xedges > psi_ctr_1[i])[0][0] - 1
        if phi_ctr_1[i] > 179.0:
            index_y = np.where(yedges > 179.0)[0][0] - 1
        else:
            index_y = np.where(yedges > phi_ctr_1[i])[0][0] - 1

        index_distrib = int(H[index_x][index_y])
        distribution_1[index_distrib] += 1
    distribution_1 /= len(phi_ctr_1)
    print(distribution_1)

    distribution_2 = np.array([0,0,0,0,0,0,0,0,0,0], dtype=np.float64)
    for i in xrange(0, len(phi_ctr_2)):
        if psi_ctr_2[i] > 179.0:
            index_x = np.where(xedges > 179.0)[0][0] - 1
        else:
            index_x = np.where(xedges > psi_ctr_2[i])[0][0] - 1
        if phi_ctr_2[i] > 179.0:
            index_y = np.where(yedges > 179.0)[0][0] - 1
        else:
            index_y = np.where(yedges > phi_ctr_2[i])[0][0] - 1

        index_distrib = int(H[index_x][index_y])
        distribution_2[index_distrib] += 1
    distribution_2 /= len(phi_ctr_2)
    print(distribution_2)
       # print "clenter:", i, "[", phi_ctr,",", psi_ctr,"]", "H=", H[index_x][index_y]
    plt.xlabel('$\phi$', fontsize=20)
    plt.ylabel('$\Psi$', fontsize=20)

    #cbar = plt.colorbar(shrink=0.77)
    ##plt.title('Free energy landscape', fontsize=20)
    #cbar.set_label("$k_B T$", size=20)
    #cbar.ax.tick_params(labelsize=20)
    #if potential is False:
    #    plt.xlim([-180, 180])
    #    plt.ylim([-180, 180])
    #    plt.xticks([-120, -60, 0, 60, 120])
    #    plt.yticks([-120, -60, 0, 60, 120])
    #else:
    #    plt.xlim([-75, 75])
    #    plt.ylim([-75, 75])
    #    plt.xticks([-50, 0, 50])
    #    plt.yticks([-50, 0, 50])
    #plt.savefig('./' + name + '.png', dpi=400)
    ##plt.show()
    #plt.close()

    #Cluster Centers on Free energy landscape distribution
    fig=plt.figure(1, (10,6))
    plt.rc("font", size=15)
    distrib = fig.add_subplot(1,1,1)
    distrib.grid(True)
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    distrib.yaxis.set_major_formatter(xticks)
    # plt.xticks(np.arange(11), ('', '1', '', '3', '', '5', '', '7', '', '9', ''))
    n_groups = 10
    index = np.arange(n_groups)
    bar_width = 0.45
    distrib.bar(index, distribution_1*100, facecolor='black', edgecolor='white', width=bar_width, label="kNN Density Peaks 3645 states") #facecolor='#f78181'
    distrib.bar(index+bar_width, distribution_2*100, facecolor='gray', edgecolor='white', width=bar_width, label="kNN Density Peaks 117 states")

    #plt.title('Cluster Centers on Free energy landscape distribution', fontsize=10)
    plt.xlabel("$k_B T$")
    plt.ylabel(r"Fraction number of clusters")
    plt.ylim([0, 50])
    plt.xticks(index+bar_width, ('', '1', '', '3', '', '5', '', '7', '', '9', ''))
    plt.legend()
    #plt.tight_layout()
    plt.savefig('./' + name + '_Distribution.png', dpi=400)
    #plt.show()
    plt.close()

def plot_landscape_barrier(labels=None, selected=1, phi_angles=None, psi_angles=None, phi_ctr=None, psi_ctr=None, name='Energy_Landscape', bins=36, potential=False, outliers=-1):
    H, xedges, yedges = np.histogram2d(psi_angles, phi_angles, bins=bins)
    #since we calculate total number in 10 interval, thus bin of every dimension must be 36
    #If element in H is zero, set the final energy to be 9
    plt.rc("font", size=25)
    maxH = np.max(H)
    for i in range(len(H)):
        for j in range(len(H)):
            if H[i][j]==0:
                H[i][j]=9
            else:
                H[i][j] = -np.log(H[i][j]/maxH)

    #H = -np.log(H/np.max(H))
    extent =[np.min(xedges), np.max(xedges), np.min(yedges), np.max(yedges)]
    plt.figure(figsize=(12, 12))
    plt.imshow(H, extent=extent, origin="lower", cmap=plt.cm.gray)  #plt.cm.jet

    #plot points
    colors = ['y', 'b', 'tomato', 'm', 'g', 'c', 'yellowgreen']
    color_index = 0
    clusters = np.unique(labels)
    for i in clusters:
        if i != outliers:
            if i in selected:
                point = np.where(labels == i)
                plt.plot(phi_angles[point], psi_angles[point], '2',  alpha=0.20, color=colors[color_index])#, color=colors_jet[i])
                color_index += 1
    #plot cluster centers on landscape
    if labels is not None:
        plt.plot(phi_ctr, psi_ctr, '*', markersize=10, color='r')

    distribution = np.array([0,0,0,0,0,0,0,0,0,0], dtype=np.float64)
    #print "len phi_ctr", len(phi_ctr)
    #print "shape of xedges", xedges.shape
    for i in xrange(0, len(phi_ctr)):
        if psi_ctr[i] > 179.0:
            index_x = np.where(xedges > 179.0)[0][0] - 1
        else:
            index_x = np.where(xedges > psi_ctr[i])[0][0] - 1
        if phi_ctr[i] > 179.0:
            index_y = np.where(yedges > 179.0)[0][0] - 1
        else:
            index_y = np.where(yedges > phi_ctr[i])[0][0] - 1

        index_distrib = int(H[index_x][index_y])
        distribution[index_distrib] += 1
    distribution /= len(phi_ctr)
    print(distribution)
       # print "clenter:", i, "[", phi_ctr,",", psi_ctr,"]", "H=", H[index_x][index_y]
    plt.xlabel('$\phi$', fontsize=20)
    plt.ylabel('$\Psi$', fontsize=20)

    cbar = plt.colorbar(shrink=0.77)
    #plt.title('Free energy landscape', fontsize=20)
    cbar.set_label("$k_B T$", size=20)
    cbar.ax.tick_params(labelsize=20)
    plt.xlim([-180, 180])
    plt.ylim([-180, 180])
    plt.xticks([-120, -60, 0, 60, 120])
    plt.yticks([-120, -60, 0, 60, 120])
    plt.plot([-103,-103],[30,180],'w') #plot the barrier
    plt.savefig('./' + name + '.png', dpi=400)
    #plt.show()
    plt.close()

def calculate_population(labels, name='Populations'):
    print("Calculating and plotting population...")
    counts = list(Counter(labels).values())
    total_states = np.max(labels) + 1
    #states_magnitude = int(np.ceil(np.log10(total_states)))
    total_frames = len(labels)
    frames_magnitude = int(np.ceil(np.log10(total_frames)))
    print("states", total_states, "frames", total_frames)

    populations = np.zeros(frames_magnitude+1)
    for i in counts:
        if i > 0:
            log_i = np.log10(i)
            magnitude = np.ceil(log_i)
            populations[magnitude] += 1

    #print magnitude populations
    print("Populations Probability:")
    #bins = [0]
    for i in xrange(len(populations)):
        populations[i] = populations[i] / total_states
        print("10 ^", i, "to", "10 ^", i+1,":", populations[i]*100, "%")
        #bins.append(10**(i+1))

    name += '_Populations'
    print("name:", name)
    plot_cluster_size_distribution(populations=populations, name=name)
    print("Done.")

def compare_population(labels_1, labels_2, name='Compare_Populations'):
    print("Calculating and plotting population...")
    counts = list(Counter(labels_1).values())
    total_states = np.max(labels_1) + 1
    total_frames = len(labels_1)
    frames_magnitude = int(np.ceil(np.log10(total_frames)))
    print("states", total_states, "frames", total_frames)

    populations_1 = np.zeros(frames_magnitude+1)
    for i in counts:
        if i > 0:
            log_i = np.log10(i)
            magnitude = np.ceil(log_i)
            populations_1[magnitude] += 1

    print("Populations Probability:")
    for i in xrange(len(populations_1)):
        populations_1[i] = populations_1[i] / total_states
        print("10 ^", i, "to", "10 ^", i+1,":", populations_1[i]*100, "%")


    counts = list(Counter(labels_2).values())
    total_states = np.max(labels_2) + 1
    total_frames = len(labels_2)
    frames_magnitude = int(np.ceil(np.log10(total_frames)))
    print("states", total_states, "frames", total_frames)

    populations_2 = np.zeros(frames_magnitude+1)
    for i in counts:
        if i > 0:
            log_i = np.log10(i)
            magnitude = np.ceil(log_i)
            populations_2[magnitude] += 1

    print("Populations Probability:")
    for i in xrange(len(populations_2)):
        populations_2[i] = populations_2[i] / total_states
        print("10 ^", i, "to", "10 ^", i+1,":", populations_2[i]*100, "%")

    name += '_Populations'
    print("name:", name)
    plot_compare_cluster_size_distribution(populations_1=populations_1, populations_2=populations_2, name=name)
    #plot_cluster_size_distribution(populations_1=populations_1, name=name)
    print("Done.")

def calculate_landscape(labels, centers, phi_angles, psi_angles, potential=False, name='Energy_Landscape'):
    print("Calculating and plotting Landscape...")
    phi_ctr = phi_angles[centers]
    psi_ctr = psi_angles[centers]
    labels_ctr = labels[centers]

    name = name + '_Energy_Landscape'
    print("name:", name)
    plot_landscape(labels=labels_ctr, phi_angles=phi_angles, psi_angles=psi_angles, phi_ctr=phi_ctr, psi_ctr=psi_ctr, potential=potential, name=name)
    print("Done")
    #plot_landscape(labels=None, phi_angles=phi_angles, psi_angles=psi_angles)
