# %%Author Wei Wang
# plot the distribution of variance in sphere

import numpy
from sklearn.neighbors import KDTree
import matplotlib
# matplotlib.use('Agg')
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import pyemma

'''
Converting the 3D-Euler angles (in global 'zyz' rotation) into viewing directions
Having checked with my matlab code and the current code is correct
'''


def rot_dim(option, angle):  # angle should be in degree measure
    # convert degree to anti-clockwise radian
    angle_radian = -angle / 360.0 * 2 * numpy.pi
    if option == 'x':
        result = [[1, 0, 0], [0, numpy.cos(angle_radian), -numpy.sin(angle_radian)],
                  [0, numpy.sin(angle_radian), numpy.cos(angle_radian)]]
    elif option == 'y':
        result = [[numpy.cos(angle_radian), 0, numpy.sin(angle_radian)], [0, 1, 0],
                  [-numpy.sin(angle_radian), 0, numpy.cos(angle_radian)]]
    elif option == 'z':
        result = [[numpy.cos(angle_radian), -numpy.sin(angle_radian), 0],
                  [numpy.sin(angle_radian), numpy.cos(angle_radian), 0], [0, 0, 1]]
    else:
        print("please provide valid inputs")
    return numpy.matrix(result)


def Euler2Rot(option, angles):  # here the angles are in degree measure
    # we use global 'zyz' rotation in the cryo-EM project, make sure they are clockwise rotation angles.
    if option == 'zyz':
        rot = rot_dim('z', -angles[0]).dot(rot_dim('y', -angles[1])).dot(rot_dim('z', -angles[2]))
        return rot


def viewing_direction(option, angles):
    v = numpy.array([0, 0, 1])
    if option == 'zyz':
        return Euler2Rot('zyz', angles).dot(v)


# %%

folder = './'
raw_info = numpy.loadtxt(folder + 'result_collect_contributions_weighted.dat')[:, 1:]
top_PC = 3
raw_info_new = raw_info.copy()
'''temp = raw_info[:, 3]
for j in range(4, 7):
    temp = temp + raw_info[:, j]
    raw_info_new[:, j] = temp'''

data_xy = []  # because we don't care about the third euler angle for the viewing direction

# %%

print(raw_info_new[0, :])

# %%

for j in range(len(raw_info)):
    temp = viewing_direction('zyz', [raw_info_new[j][0], raw_info_new[j][1], raw_info_new[j][2]])
    if (temp[0, 2] < 0):
        data_xy.append([-temp[0, 0], -temp[0, 1]])  # reflect
    else:
        data_xy.append([temp[0, 0], temp[0, 1]])

# %%

# data_xy will be used to create the kd tree

# as the 2500+ points in my case are almost uniformaly distributed on the sphere, therefore we can simply query the nearest k neighbors rather than specify a radius
data_xy = numpy.array(data_xy)
tree = KDTree(data_xy, leaf_size=10)
new_weights = []
# query for all the points
nn = 1
dist, ind = tree.query(data_xy, k=nn)

# %%

new_weights = []
print(raw_info_new.shape)
for j in range(len(data_xy)):
    value_list = [raw_info_new[m][top_PC + 3] for m in ind[j]]
    new_weights.append(numpy.mean(value_list))

outputfile = folder + 'projection_vector_variance_knn_new_variance_of_top_%d_PC.dat' % (top_PC)
f = open(outputfile, 'w')
for j in range(len(data_xy)):
    f.write('%f %f %f %f\n' % (data_xy[j][0], data_xy[j][1], raw_info_new[j][top_PC + 2], new_weights[j]))
f.close()

# %%

data_xy = numpy.array(data_xy)
plt.scatter(data_xy[:, 0], data_xy[:, 1], s=2)
plt.tick_params(labelsize=25)


# %%

pyemma.plots.plot_free_energy(data_xy[:, 0], data_xy[:, 1], weights=new_weights, logscale=True, nbins=25)
point_xyz = viewing_direction('zyz', [0, 70, 0])
plt.scatter(point_xyz[0, 0], point_xyz[0, 1], s=1000, color='black', marker='p', alpha=0.5)

plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.tick_params(labelsize=25)

plt.savefig('top_%d_PCs_total_contributions.png' % (top_PC), dpi=300)

# %%


# %%
