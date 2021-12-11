# Writed on July 16, 2018 by Wei Wang
# Modified on Aug 31, 2020 by Hanlin Gu

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import mrcfile
import cv2
from skimage.exposure import rescale_intensity
from skimage import measure
import logging
import multiprocessing
import os
from sklearn.metrics.pairwise import pairwise_distances
import math
from skimage.segmentation import random_walker
from skimage.filters import threshold_otsu
import sh



def calculate_distance(new_test_image_all, landmark_image_id, len_landmark, len_test, landmark_image_all, angle_list,
                       rot_angle_matrices,
                       cols, rows):
    landmark_image = landmark_image_all[landmark_image_id]
    dist_list = np.zeros([len_test])
    landmark_clock = []
    landmark_anti = []
    temp_test_image_reshape = np.reshape(new_test_image_all, (len_test, cols * rows))
    for angle_id in range(len(angle_list)):
        if (angle_id != 0):
            temp_image = cv2.warpAffine(landmark_image, rot_angle_matrices[angle_id], (cols, rows),
                                        borderValue=0)  # since this is the raw images
        else:
            temp_image = landmark_image
        landmark_clock.append(temp_image)
        landmark_anti.append(cv2.flip(temp_image, 0))

    temp_clock_landmark_reshape = np.reshape(landmark_clock, (len(landmark_clock), cols * rows))
    temp_anti_landmark_reshape = np.reshape(landmark_anti, (len(landmark_anti), cols * rows))

    dist_clock = pairwise_distances(temp_test_image_reshape, temp_clock_landmark_reshape, metric='euclidean')
    dist_anti = pairwise_distances(temp_test_image_reshape, temp_anti_landmark_reshape, metric='euclidean')

    for test_image_id in range(len_test):
        if min(dist_clock[test_image_id]) < min(dist_anti[test_image_id]):
            dist_list[test_image_id] = min(dist_clock[test_image_id])
        else:
            dist_list[test_image_id] = min(dist_anti[test_image_id])
    return dist_list


def parallel_calculate_distances(new_test_image_all, new_landmark_ids, len_landmark, len_test, new_dist_,
                                 landmark_image_all, angle_list,
                                 rot_angle_matrices, cols, rows):
    lock = multiprocessing.Lock()
    # cpus = multiprocessing.cpu_count()
    cpus = 4  # change
    pool = multiprocessing.Pool(processes=cpus)
    print('Running on %d cpus' % (cpus))
    multi_results = [pool.apply_async(calculate_distance, args=(
        new_test_image_all, landmark_image_id, len_landmark, len_test, landmark_image_all, angle_list,
        rot_angle_matrices, cols, rows)) for
                     landmark_image_id in new_landmark_ids]

    index = 0
    for res in multi_results:
        print("begin to calculating for landmark %d" % (index))
        landmark_image_id = new_landmark_ids[index]
        new_dist_[:, landmark_image_id] = res.get()
        index += 1

    # for landmark_image_id in xrange(len_landmark):
    #	if landmark_image_id in new_landmark_ids:
    #		print('now begin to focus on landmark:%d'%(landmark_image_id))
    #		dist_list=pool.apply_async(calculate_distance, args=(landmark_image_id, len_landmark, new_test_image_all, len_test, landmark_image_all, angle_list, rot_angle_matrices, cols, rows))
    #		new_dist_[:, landmark_image_id] = dist_list.get()
    pool.close()


###FOR CALCULATING ACT FUNCTIONS
def normcc(x1, y1, x2, y2,
           dim):  # not normalized correlation function between two vectors, x1, y1 are x part and y part of vector 1
    cross = 0
    for j in range(dim):
        cross = cross + (x1[j] * x2[j] + y1[j] * y2[j])
    #        cross = cross+np.sqrt((x1[j]*x2[j]+y1[j]*y2[j])**2+(x2[j]*y1[j]-x1[j]*y2[j])**2)
    auto1 = 0
    auto2 = 0
    for j in range(dim):
        auto1 = auto1 + (x1[j] * x1[j] + y1[j] * y1[j])
        auto2 = auto2 + (x2[j] * x2[j] + y2[j] * y2[j])
    #    return cross/(np.sqrt(auto1)*np.sqrt(auto2))
    return cross


def ACF(vector_x, vector_y):
    ACF = []
    for j in range(int(vector_x.shape[0] / 2)):
        x1 = vector_x
        y1 = vector_y
        x2 = np.roll(x1, j)
        y2 = np.roll(y1, j)
        ACF.append(normcc(x1, y1, x2, y2, vector_x.shape[0]))
    return ACF


def perform_ACF_for_batch_images(batch_name):
    ACF_all = []
    for img in batch_name:
        contours = measure.find_contours(img, 0.8)  # here we have the parameter
        maxContour = max(contours, key=len)
        n_points = 100  # here we have a parameter, # of boundary points to extract
        # we only keep the maximum contour
        step = maxContour.shape[0] / float(n_points)
        indices = np.arange(n_points) * step
        indices = np.around(indices)
        indices = indices.astype(int)
        maxContour = maxContour[indices]
        maxContour = maxContour.astype(int)
        similar = maxContour
        vector_similar_x = np.ediff1d(similar[:, 0], to_end=(similar[0, 0] - similar[-1, 0]))
        vector_similar_y = np.ediff1d(similar[:, 1], to_end=(similar[0, 1] - similar[-1, 1]))
        ACF_all.append(ACF(vector_similar_x, vector_similar_y))
    return ACF_all


def min_row_ndarray(input_matrix):
    if (len(input_matrix.shape) == 1):
        return input_matrix
    else:
        return np.min(input_matrix, 1)


def oneNN_predict(distance_matrix, labels_, labels_sub):
    array = []
    for j in range(len(labels_sub)):
        array_temp = np.min(distance_matrix[:, labels_sub[j]], axis=1)  # row
        array.append(array_temp)
    array_temp = np.min(distance_matrix, axis=1)
    array.append(array_temp)

    array_temp = np.argmin(distance_matrix, axis=1)  # for each data, overall minimum

    array.append(labels_[array_temp])  # overall
    result = []
    for j in range(len(array_temp)):  # number of data
        temp_row = []
        for k in range(len(array)):
            temp_row.append(array[k][j])
        result.append(temp_row)  # maybe should use concatenate here
    return result


def get_labels_sub(labels_):  # suppose they start from 0
    labels_sub = []
    for j in range(min(labels_), max(labels_) + 1):
        index = np.where(labels_ == j)[0]
        labels_sub.append(index)
    return labels_sub


def findGoldMask(data):
    thresh = threshold_otsu(data)
    sigma1 = np.std(data[np.where(data < thresh)])
    sigma2 = np.std(data[np.where(data > thresh)])
    markers = np.zeros(data.shape, dtype=np.uint)
    markers[data < thresh - 0.5 * sigma1] = 1
    markers[data > thresh + 0.5 * sigma2] = 2
    labels = random_walker(data, markers, beta=10, mode='bf')
    labels[labels != 2] = 0
    labels[labels == 2] = 1
    return labels


def two_stage_matching(datatype, path4):
    logging.getLogger().setLevel(logging.INFO)
    resultdir = path4   # need to revise

    ####write landmark   here we take sim as an example,for real data, you need to replace the open, close and inter wit H_EV2_red...
    template_anlge = np.loadtxt('./angles_for_centers.txt')
    focus = np.loadtxt('./gen_2D_image/angles_for_rotate_z_y_z_new.txt')
    f = open(path4 + 'landmark_used_in_1st_stage.list', 'w')
    for i in range(template_anlge.shape[0]):
        f.write('./data/templates_' + 'real_for_zyz_')
        f.write(str(focus[0]) + '_' + str(focus[1]) + '_' + str(focus[2]))
        f.write('/open_zyz_')
        f.write(str(focus[0]) + '_' + str(focus[1]) + '_' + str(focus[2]) + '_')
        f.write('rotate_')
        f.write(str(int(template_anlge[i][0])) + '_' + str(int(template_anlge[i][1])) + '_' + str(
            int(template_anlge[i][2])))
        f.write('.mrc')
        f.write('\n')
                f.write('./data/templates_' + 'real_for_zyz_')
        f.write(str(focus[0]) + '_' + str(focus[1]) + '_' + str(focus[2]))
        f.write('/intermediate_zyz_')
        f.write(str(focus[0]) + '_' + str(focus[1]) + '_' + str(focus[2]) + '_')
        f.write('rotate_')
        f.write(str(int(template_anlge[i][0])) + '_' + str(int(template_anlge[i][1])) + '_' + str(
            int(template_anlge[i][2])))
        f.write('.mrc')
        f.write('\n')
                f.write('./data/templates_' + 'real_for_zyz_')
        f.write(str(focus[0]) + '_' + str(focus[1]) + '_' + str(focus[2]))
        f.write('/close_zyz_')
        f.write(str(focus[0]) + '_' + str(focus[1]) + '_' + str(focus[2]) + '_')
        f.write('rotate_')
        f.write(str(int(template_anlge[i][0])) + '_' + str(int(template_anlge[i][1])) + '_' + str(
            int(template_anlge[i][2])))
        f.write('.mrc')
        f.write('\n')
    f.close()
    sh.bash(path4 + 'gen_landmark.sh')  ####here generate the landmark which represents the three color range same as paper.

    # create formatter
    logging.basicConfig(format='%(asctime)s %(message)s')
    ####reading in the input images and landmark images#############

    logging.info('start of the program')

    logging.info("reading in the input images in stack")

    # the input datasets
    # resultdir, test_data_filelist.list, landmarks_all.txt
    # landmarks_1st_labels_, landmarks_2nd_labels_

    ##things to do: write the options (arguments)

    # Part I: Generate ACFs (because they will be widely applied later to filter the majority of conformations)

    # reading the reference trajectories list
    test_image_batch_name = []
    for line in open(resultdir + datatype + 'test_data_filelist.list'):  # input
        test_image_batch_name.append(line.strip())

    # get the ACF for the test images
    for line in test_image_batch_name:
        if os.path.isfile(
                resultdir + 'ACF_of_range' + datatype + '_' + os.path.basename(line)[:-5] + '.npy'):
            continue
        elif line != '':
            logging.info("generating ACFs for the dataset %s" % (line))
            # generate the ACF for these images
            temp = mrcfile.open(line).data
            temp_stack_masked = []
            for temp_id in range(len(temp)):
                image_temp = temp[temp_id]
                temp_stack_masked.append(
                    findGoldMask(rescale_intensity(1.0 * image_temp, out_range=(0, 1))))  # mask the images
            ACF_temp_stack = perform_ACF_for_batch_images(temp_stack_masked)
            np.save(
                resultdir + 'ACF_of_range' + datatype + '_' + os.path.basename(line)[:-5] + '.npy',
                ACF_temp_stack)
            del temp_stack_masked, ACF_temp_stack

    # reading the template images trajectory list
    landmark_name = resultdir + 'landmark_used_in_1st_stage.list'  # input
    # get the ACF for the template images
    if os.path.isfile(resultdir + 'ACF_of_range' + datatype + '_template_images.npy'):
        print("we don't need to regenerate ACF")
    else:
        logging.info("generating ACFs for the template images")
        temp_stack_masked = []
        for line in open(landmark_name):
            temp = mrcfile.open(line.strip()).data[0]  # not stack here
            temp_stack_masked.append(findGoldMask(rescale_intensity(1.0 * temp, out_range=(0, 1))))  # mask
        ACF_temp_stack = perform_ACF_for_batch_images(temp_stack_masked)
        np.save(resultdir + 'ACF_of_range' + datatype + '_template_images.npy', ACF_temp_stack)
        del temp_stack_masked, ACF_temp_stack

    #	for the 1st stage classification, angular classification based on 1nn
    landmarks_1st_labels_ = np.loadtxt(resultdir + 'landmarks_used_in_1st_stage.labels_', dtype=int)
    ACF_labels_sub = get_labels_sub(landmarks_1st_labels_)

    #	reading in the ACFs for the template images
    logging.info("reading in the ACFs for the landmark images and perform 1st classifying")

    ACF_landmark_image_all = np.load('ACF_of_range' + datatype + '_template_images.npy')

    logging.info('begin calculating the ACF pairwise distances')

    metric_option = 'euclidean'
    image_id_for_each_stack_go_to_second_round = []
    ACF_result = open(resultdir + 'ACF_classification_result_%s' % (metric_option), 'w')

    for line in test_image_batch_name:
        if line != '':
            temp_id_stack = []
            ACF_test_image_all = np.load(
                resultdir + 'ACF_of_range' + datatype + '_' + os.path.basename(line)[:-5] + '.npy')
            ACF_dist = pairwise_distances(ACF_test_image_all, ACF_landmark_image_all, metric=metric_option)
            # no need to save them as they are easy to be computed
            output1 = oneNN_predict(ACF_dist, landmarks_1st_labels_, ACF_labels_sub)
            for temp_id in range(len(ACF_test_image_all)):
                if (output1[temp_id][-1] == 1):  # what we want to keep
                    temp_id_stack.append(temp_id)  # starting from 0
                ACF_result.write("%s@%s  %s  \n" % (format(temp_id + 1, '06d'), line, output1[temp_id]))
            image_id_for_each_stack_go_to_second_round.append(temp_id_stack)

    print(image_id_for_each_stack_go_to_second_round[0])
    # image_id_for_each_stack_go_to_second_round = np.array(image_id_for_each_stack_go_to_second_round)
    del ACF_test_image_all, ACF_dist, ACF_landmark_image_all
    ################################## Now we calculate the minimum distance of each conformation to self.interest, self.notinterest and inter###########

    logging.info('summarize the ACF classification results')

    ACF_result.close()

    ################################ summarize the statistics of the results:run post_plot.py #################################
    logging.info("now begin to get the statistics of ACF classification results")

    print("the number of images that go to 2nd round is:",
          len(np.concatenate(image_id_for_each_stack_go_to_second_round)))
    logging.info(
        'extracting the images assigned to self interest landmarks, and preparing for 2nd stage classification, it would take even longer time...')

    # 2nd stage classification based on grayscale images

    # loading the landmark trajectories
    landmark_name_new = resultdir + 'landmark_used_in_2nd_stage.list'
    landmarks_2nd_labels_ = np.loadtxt(resultdir + 'landmarks_used_in_2nd_stage.labels_', dtype=int)

    # we need to remove the -1 clusters (the ones we are not interested in)
    grayscale_labels_sub = get_labels_sub(landmarks_2nd_labels_)  # from 0

    logging.info("loading the landmark images that are used in 2nd stage and normalize them")
    landmark_image_all_raw = []
    for line in open(landmark_name_new):
        temp = mrcfile.open(line.strip()).data[0]
        landmark_image_all_raw.append(
            rescale_intensity(1.0 * temp, out_range=(0, 1)))  # normalize the images   ##template image

    logging.info("loading the test images and perform the 2nd classification")

    new_test_image_all = []

    for j in range(len(image_id_for_each_stack_go_to_second_round)):
        temp = mrcfile.open(test_image_batch_name[j]).data
        temp_subset = temp[image_id_for_each_stack_go_to_second_round[j]]
        for temp_id in range(len(image_id_for_each_stack_go_to_second_round[j])):
            temp_normalized = temp_subset[temp_id]
            new_test_image_all.append(rescale_intensity(1.0 * temp_normalized, out_range=(0, 1)))  ###test image
    #	new_test_image_all = np.array(new_test_image_all)
    #	landmark_image_all_raw = np.array(landmark_image_all_raw)

    angle_scan_rate = 1
    # shift_x_scan = 5;
    # shift_y_scan = 5;

    angle_list = range(0, 360, angle_scan_rate)
    len_test = len(new_test_image_all)
    len_landmark = len(landmark_image_all_raw)
    cols, rows = landmark_image_all_raw[0].shape

    new_dist_ = np.zeros([len_test, len_landmark])
    rot_angle_matrices = []
    for rot_angle in angle_list:
        rot_angle_matrices.append(cv2.getRotationMatrix2D((cols / 2, rows / 2), rot_angle, 1))  # center, angle, scale

    new_landmark_ids = range(len_landmark)
    parallel_calculate_distances(new_test_image_all, new_landmark_ids, len_landmark, len_test, new_dist_,
                                 landmark_image_all_raw,
                                 angle_list, rot_angle_matrices, cols, rows)
    print(new_dist_.shape)
    output2 = oneNN_predict(new_dist_, landmarks_2nd_labels_, grayscale_labels_sub)

    logging.info('end of the 2nd stage classification and results are saved')

    temp_len = 0
    second_round_result = open(resultdir + datatype + '2nd_stage_brute_force_matching_result_%s' % (metric_option),
                               'w')

    logging.info('summarize the results of 2nd stage classification')

    temp_id = 0

    for j in range(len(image_id_for_each_stack_go_to_second_round)):
        for k in range(len(image_id_for_each_stack_go_to_second_round[j])):
            number = image_id_for_each_stack_go_to_second_round[j][k] + 1
            second_round_result.write(
                "%s@%s  %s\n" % (format(number, '06d'), test_image_batch_name[j], output2[temp_id]))
            temp_id = temp_id + 1

    second_round_result.close()

    logging.info('end of the program')


if __name__ == '__main__':
    main()



def calculate_distance(new_test_image_all, landmark_image_id, len_landmark, len_test, landmark_image_all, angle_list,
                       rot_angle_matrices,
                       cols, rows):
    landmark_image = landmark_image_all[landmark_image_id]
    dist_list = np.zeros([len_test])
    landmark_clock = []
    landmark_anti = []
    temp_test_image_reshape = np.reshape(new_test_image_all, (len_test, cols * rows))
    for angle_id in range(len(angle_list)):
        if (angle_id != 0):
            temp_image = cv2.warpAffine(landmark_image, rot_angle_matrices[angle_id], (cols, rows),
                                        borderValue=0)  # since this is the raw images
        else:
            temp_image = landmark_image
        landmark_clock.append(temp_image)
        landmark_anti.append(cv2.flip(temp_image, 0))

    temp_clock_landmark_reshape = np.reshape(landmark_clock, (len(landmark_clock), cols * rows))
    temp_anti_landmark_reshape = np.reshape(landmark_anti, (len(landmark_anti), cols * rows))

    dist_clock = pairwise_distances(temp_test_image_reshape, temp_clock_landmark_reshape, metric='euclidean')
    dist_anti = pairwise_distances(temp_test_image_reshape, temp_anti_landmark_reshape, metric='euclidean')

    for test_image_id in range(len_test):
        if min(dist_clock[test_image_id]) < min(dist_anti[test_image_id]):
            dist_list[test_image_id] = min(dist_clock[test_image_id])
        else:
            dist_list[test_image_id] = min(dist_anti[test_image_id])
    return dist_list


def parallel_calculate_distances(new_test_image_all, new_landmark_ids, len_landmark, len_test, new_dist_,
                                 landmark_image_all, angle_list,
                                 rot_angle_matrices, cols, rows):
    lock = multiprocessing.Lock()
    # cpus = multiprocessing.cpu_count()
    cpus = 4  # change
    pool = multiprocessing.Pool(processes=cpus)
    print('Running on %d cpus' % (cpus))
    multi_results = [pool.apply_async(calculate_distance, args=(
        new_test_image_all, landmark_image_id, len_landmark, len_test, landmark_image_all, angle_list,
        rot_angle_matrices, cols, rows)) for
                     landmark_image_id in new_landmark_ids]

    index = 0
    for res in multi_results:
        print("begin to calculating for landmark %d" % (index))
        landmark_image_id = new_landmark_ids[index]
        new_dist_[:, landmark_image_id] = res.get()
        index += 1

    # for landmark_image_id in xrange(len_landmark):
    #	if landmark_image_id in new_landmark_ids:
    #		print('now begin to focus on landmark:%d'%(landmark_image_id))
    #		dist_list=pool.apply_async(calculate_distance, args=(landmark_image_id, len_landmark, new_test_image_all, len_test, landmark_image_all, angle_list, rot_angle_matrices, cols, rows))
    #		new_dist_[:, landmark_image_id] = dist_list.get()
    pool.close()


###FOR CALCULATING ACT FUNCTIONS
def normcc(x1, y1, x2, y2,
           dim):  # not normalized correlation function between two vectors, x1, y1 are x part and y part of vector 1
    cross = 0
    for j in range(dim):
        cross = cross + (x1[j] * x2[j] + y1[j] * y2[j])
    #        cross = cross+np.sqrt((x1[j]*x2[j]+y1[j]*y2[j])**2+(x2[j]*y1[j]-x1[j]*y2[j])**2)
    auto1 = 0
    auto2 = 0
    for j in range(dim):
        auto1 = auto1 + (x1[j] * x1[j] + y1[j] * y1[j])
        auto2 = auto2 + (x2[j] * x2[j] + y2[j] * y2[j])
    #    return cross/(np.sqrt(auto1)*np.sqrt(auto2))
    return cross


def ACF(vector_x, vector_y):
    ACF = []
    for j in range(int(vector_x.shape[0] / 2)):
        x1 = vector_x
        y1 = vector_y
        x2 = np.roll(x1, j)
        y2 = np.roll(y1, j)
        ACF.append(normcc(x1, y1, x2, y2, vector_x.shape[0]))
    return ACF


def perform_ACF_for_batch_images(batch_name):
    ACF_all = []
    for img in batch_name:
        contours = measure.find_contours(img, 0.8)  # here we have the parameter
        maxContour = max(contours, key=len)
        n_points = 100  # here we have a parameter, # of boundary points to extract
        # we only keep the maximum contour
        step = maxContour.shape[0] / float(n_points)
        indices = np.arange(n_points) * step
        indices = np.around(indices)
        indices = indices.astype(int)
        maxContour = maxContour[indices]
        maxContour = maxContour.astype(int)
        similar = maxContour
        vector_similar_x = np.ediff1d(similar[:, 0], to_end=(similar[0, 0] - similar[-1, 0]))
        vector_similar_y = np.ediff1d(similar[:, 1], to_end=(similar[0, 1] - similar[-1, 1]))
        ACF_all.append(ACF(vector_similar_x, vector_similar_y))
    return ACF_all


def min_row_ndarray(input_matrix):
    if (len(input_matrix.shape) == 1):
        return input_matrix
    else:
        return np.min(input_matrix, 1)


def oneNN_predict(distance_matrix, labels_, labels_sub):
    array = []
    for j in range(len(labels_sub)):
        array_temp = np.min(distance_matrix[:, labels_sub[j]], axis=1)  # row
        array.append(array_temp)
    array_temp = np.min(distance_matrix, axis=1)
    array.append(array_temp)

    array_temp = np.argmin(distance_matrix, axis=1)  # for each data, overall minimum

    array.append(labels_[array_temp])  # overall
    result = []
    for j in range(len(array_temp)):  # number of data
        temp_row = []
        for k in range(len(array)):
            temp_row.append(array[k][j])
        result.append(temp_row)  # maybe should use concatenate here
    return result


def get_labels_sub(labels_):  # suppose they start from 0
    labels_sub = []
    for j in range(min(labels_), max(labels_) + 1):
        index = np.where(labels_ == j)[0]
        labels_sub.append(index)
    return labels_sub


def findGoldMask(data):
    thresh = threshold_otsu(data)
    sigma1 = np.std(data[np.where(data < thresh)])
    sigma2 = np.std(data[np.where(data > thresh)])
    markers = np.zeros(data.shape, dtype=np.uint)
    markers[data < thresh - 0.5 * sigma1] = 1
    markers[data > thresh + 0.5 * sigma2] = 2
    labels = random_walker(data, markers, beta=10, mode='bf')
    labels[labels != 2] = 0
    labels[labels == 2] = 1
    return labels


def main():
    logging.getLogger().setLevel(logging.INFO)
    n = '1'  #1st experiment
    method_name = 'wgan'

    resultdir = './'  # need to revise
    resultdir1 = './temp_range' + str(n) + '/'
    # create formatter
    logging.basicConfig(format='%(asctime)s %(message)s')
    ####reading in the input images and landmark images#############

    logging.info('start of the program')

    logging.info("reading in the input images in stack")

    # the input datasets
    # resultdir, test_data_filelist.list, landmarks_all.txt
    # landmarks_1st_labels_, landmarks_2nd_labels_

    ##things to do: write the options (arguments)

    # Part I: Generate ACFs (because they will be widely applied later to filter the majority of conformations)

    # reading the reference trajectories list
    test_image_batch_name = []
    for line in open(resultdir + 'test_data_filelist.list'):  # input
        test_image_batch_name.append(line.strip())

    # get the ACF for the test images
    for line in test_image_batch_name:
        if os.path.isfile(
                resultdir + 'ACF_of_range' + str(n) + method_name + '_' + os.path.basename(line)[:-5] + '.npy'):
            continue
        elif line != '':
            logging.info("generating ACFs for the dataset %s" % (line))
            # generate the ACF for these images
            temp = mrcfile.open(line).data
            temp_stack_masked = []
            for temp_id in range(len(temp)):
                image_temp = temp[temp_id]
                temp_stack_masked.append(
                    findGoldMask(rescale_intensity(1.0 * image_temp, out_range=(0, 1))))  # mask the images
            ACF_temp_stack = perform_ACF_for_batch_images(temp_stack_masked)
            np.save(resultdir + 'ACF_of_range' + str(n) + method_name + '_' + os.path.basename(line)[:-5] + '.npy',
                    ACF_temp_stack)
            del temp_stack_masked, ACF_temp_stack

    # reading the template images trajectory list
    landmark_name = resultdir1 + 'landmark_used_in_1st_stage.list'  # input
    # get the ACF for the template images
    if os.path.isfile(resultdir + 'ACF_of_range' + str(n) + method_name + '_template_images.npy'):
        print("we don't need to regenerate ACF")
    else:
        logging.info("generating ACFs for the template images")
        temp_stack_masked = []
        for line in open(landmark_name):
            temp = mrcfile.open(line.strip()).data[0]  # not stack here
            temp_stack_masked.append(findGoldMask(rescale_intensity(1.0 * temp, out_range=(0, 1))))  # mask
        ACF_temp_stack = perform_ACF_for_batch_images(temp_stack_masked)
        np.save(resultdir + 'ACF_of_range' + str(n) + method_name + '_template_images.npy', ACF_temp_stack)
        del temp_stack_masked, ACF_temp_stack

    #	for the 1st stage classification, angular classification based on 1nn
    landmarks_1st_labels_ = np.loadtxt(resultdir1 + 'landmarks_used_in_1st_stage.labels_', dtype=int)
    ACF_labels_sub = get_labels_sub(landmarks_1st_labels_)

    #	reading in the ACFs for the template images
    logging.info("reading in the ACFs for the landmark images and perform 1st classifying")

    ACF_landmark_image_all = np.load('ACF_of_range' + str(n) + method_name + '_template_images.npy')

    logging.info('begin calculating the ACF pairwise distances')

    metric_option = 'euclidean'
    image_id_for_each_stack_go_to_second_round = []
    ACF_result = open(resultdir + 'ACF_classification_result_%s' % (metric_option), 'w')

    for line in test_image_batch_name:
        if line != '':
            temp_id_stack = []
            ACF_test_image_all = np.load(
                resultdir + 'ACF_of_range' + str(n) + method_name + '_' + os.path.basename(line)[:-5] + '.npy')
            ACF_dist = pairwise_distances(ACF_test_image_all, ACF_landmark_image_all, metric=metric_option)
            # no need to save them as they are easy to be computed
            output1 = oneNN_predict(ACF_dist, landmarks_1st_labels_, ACF_labels_sub)
            for temp_id in range(len(ACF_test_image_all)):
                if (output1[temp_id][-1] == 1):  # what we want to keep
                    temp_id_stack.append(temp_id)  # starting from 0
                ACF_result.write("%s@%s  %s  \n" % (format(temp_id + 1, '06d'), line, output1[temp_id]))
            image_id_for_each_stack_go_to_second_round.append(temp_id_stack)

    print(image_id_for_each_stack_go_to_second_round[0])
    # image_id_for_each_stack_go_to_second_round = np.array(image_id_for_each_stack_go_to_second_round)
    del ACF_test_image_all, ACF_dist, ACF_landmark_image_all
    ################################## Now we calculate the minimum distance of each conformation to self.interest, self.notinterest and inter###########

    logging.info('summarize the ACF classification results')

    ACF_result.close()

    ################################ summarize the statistics of the results:run post_plot.py #################################
    logging.info("now begin to get the statistics of ACF classification results")

    print("the number of images that go to 2nd round is:",
          len(np.concatenate(image_id_for_each_stack_go_to_second_round)))
    logging.info(
        'extracting the images assigned to self interest landmarks, and preparing for 2nd stage classification, it would take even longer time...')

    # 2nd stage classification based on grayscale images

    # loading the landmark trajectories
    landmark_name_new = resultdir1 + 'landmark_used_in_2nd_stage.list'
    landmarks_2nd_labels_ = np.loadtxt(resultdir1 + 'landmarks_used_in_2nd_stage.labels_', dtype=int)

    # we need to remove the -1 clusters (the ones we are not interested in)
    grayscale_labels_sub = get_labels_sub(landmarks_2nd_labels_)  # from 0

    logging.info("loading the landmark images that are used in 2nd stage and normalize them")
    landmark_image_all_raw = []
    for line in open(landmark_name_new):
        temp = mrcfile.open(line.strip()).data[0]
        landmark_image_all_raw.append(
            rescale_intensity(1.0 * temp, out_range=(0, 1)))  # normalize the images   ##template image

    logging.info("loading the test images and perform the 2nd classification")

    new_test_image_all = []

    for j in range(len(image_id_for_each_stack_go_to_second_round)):
        temp = mrcfile.open(test_image_batch_name[j]).data
        temp_subset = temp[image_id_for_each_stack_go_to_second_round[j]]
        for temp_id in range(len(image_id_for_each_stack_go_to_second_round[j])):
            temp_normalized = temp_subset[temp_id]
            new_test_image_all.append(rescale_intensity(1.0 * temp_normalized, out_range=(0, 1)))  ###test image
    #	new_test_image_all = np.array(new_test_image_all)
    #	landmark_image_all_raw = np.array(landmark_image_all_raw)

    angle_scan_rate = 1
    # shift_x_scan = 5;
    # shift_y_scan = 5;

    angle_list = range(0, 360, angle_scan_rate)
    len_test = len(new_test_image_all)
    len_landmark = len(landmark_image_all_raw)
    cols, rows = landmark_image_all_raw[0].shape

    new_dist_ = np.zeros([len_test, len_landmark])
    rot_angle_matrices = []
    for rot_angle in angle_list:
        rot_angle_matrices.append(cv2.getRotationMatrix2D((cols / 2, rows / 2), rot_angle, 1))  # center, angle, scale

    new_landmark_ids = range(len_landmark)
    parallel_calculate_distances(new_test_image_all, new_landmark_ids, len_landmark, len_test, new_dist_,
                                 landmark_image_all_raw,
                                 angle_list, rot_angle_matrices, cols, rows)
    print(new_dist_.shape)
    output2 = oneNN_predict(new_dist_, landmarks_2nd_labels_, grayscale_labels_sub)

    logging.info('end of the 2nd stage classification and results are saved')

    temp_len = 0
    second_round_result = open(resultdir + '2nd_stage_brute_force_classification_result_%s' % (metric_option), 'w')

    logging.info('summarize the results of 2nd stage classification')

    temp_id = 0

    for j in range(len(image_id_for_each_stack_go_to_second_round)):
        for k in range(len(image_id_for_each_stack_go_to_second_round[j])):
            number = image_id_for_each_stack_go_to_second_round[j][k] + 1
            second_round_result.write(
                "%s@%s  %s\n" % (format(number, '06d'), test_image_batch_name[j], output2[temp_id]))
            temp_id = temp_id + 1

    second_round_result.close()

    logging.info('end of the program')

