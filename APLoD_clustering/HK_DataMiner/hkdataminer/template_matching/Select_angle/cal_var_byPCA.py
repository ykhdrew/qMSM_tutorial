####edit by Hanlin Gu on 28/8/2020
####Aimed to find the best viewing anlge which can distinguish multiple conformations well
'''
----input: multiple 3D conformations: conf_size * vol_size*vol_size*vol_size
    index_angles_generating_cores : random choose angles uniformly sampled from sphere (here because euler rotation type is zyz, so we assume the third angle is 0)
----output: the variance matrix: can reflect in which viewing angle (focus) has the largest variance.
            The larger variance is, the larger distingushbilty multiple conformations have.
----focus: best viewing angle

notice:  1.we use xmipp to project 3D volume, you need to install xmipp
         2.For real data, the multiple conformations are dervied by multi-body refinement in relion, directly get PCA results, so you can ignore the PCA step here.
'''

import numpy as np
from sklearn.decomposition import PCA
import sh
import mrcfile

# training parameters


def writemrc(path_name, data, vol_size):
    with mrcfile.new(path_name) as mrc:
        mrc.set_data(np.zeros((vol_size, vol_size, vol_size), dtype=np.float32))
        mrc.data = data.reshape((vol_size, vol_size, vol_size))


class Select_anlge_step(object):

    def __init__(self, n_components=3, vol_size=128, conf_size=99, path1):
        self.n_components = n_components
        self.vol_size = vol_size
        self.conf_size = conf_size
        self.path1 = path
        self.outputname = path1 + '/variance.txt'
        
    def load_data(self, datapath):
        input = []
        for file in os.listdir(datapath):
            input.append(mrcfile.open(os.path.join(datapath, file)))
        return input

    def PCA(self, input):
        input = np.reshape(input, (self.conf_size, -1))
        pca = PCA(n_components=self.n_components).fit(input)
        pc_eigenvec = pca.components_
        pc_eigenvalue = pca.singular_values_
        for j in range(self.n_components):
            writemrc('./Select_angle/pc' + str(j + 1) + '.mrc', pc_eigenvec[j], self.vol_size)
        return pc_eigenvalue

    def project(self, pc_eigenvec):
        sh.bash('./Select_angle/project_pca.sh')

    def cal_contribution_from_each_pc_original_weighted(self, eigenvalues):
        f = open(self.outputname, 'w')  # the variance matrix
        temp_int = 1

        for line in open('./Select_angle/index_angles_generating_cores.txt'):
            line = line.strip()
            print('handling ', line)
            total = 0
            f.write('%s ' % (line))
            for j in range(1, 4, 1):
                a = mrcfile.open('./Select_angle/pc_%d_original_angle_index_%d_projection.mrc' % (j, temp_int)).data
                temp1 = np.linalg.norm(a) ** 2 * eigenvalues[j - 1]
                total += temp1
                f.write('%f ' % (temp1))
            f.write('%f \n' % (total))
            temp_int += 1

        f.close()

    def find_best_angle(self):
        data = np.loadtxt(self.outputname)
        data = data[np.lexsort(-data.T)]
        focus = [data[0][1], data[0][2], data[0][3]]
        np.savetxt('./gen_2D_image/angles_for_rotate_z_y_z_new.txt', focus)# the best anlge

