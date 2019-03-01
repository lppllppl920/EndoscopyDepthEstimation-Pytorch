'''
Author: Xingtong Liu, Ayushi Sinha, Masaru Ishii, Gregory D. Hager, Austin Reiter, Russell H. Taylor, and Mathias Unberath

Copyright (C) 2019 Johns Hopkins University - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU GENERAL PUBLIC LICENSE Version 3 license for non-commercial usage.

You should have received a copy of the GNU GENERAL PUBLIC LICENSE Version 3 license with
this file. If not, please write to: xliu89@jh.edu or rht@jhu.edu or unberath@jhu.edu
'''

import numpy as np
import cv2
from plyfile import PlyData, PlyElement
import yaml
import random
import torch
import torchvision.utils as vutils
import datetime
import shutil

import matplotlib

matplotlib.use('Agg', warn=False, force=True)
from matplotlib import pyplot as plt


def get_color_file_names_by_bag(root, which_bag, split_ratio=(0.5, 0.5)):
    training_image_list = []
    rest_image_list = []
    for i in range(1, 7):
        if i != which_bag:
            training_image_list += list(root.glob('*' + str(i) + '/_start*/0*.jpg'))
        else:
            rest_image_list = list(root.glob('*' + str(i) + '/_start*/0*.jpg'))

    training_image_list.sort()
    rest_image_list.sort()
    split_point = int(len(rest_image_list) * split_ratio[0])
    return training_image_list, rest_image_list[:split_point], rest_image_list[split_point:]


def get_color_file_names(root, split_ratio=(0.9, 0.05, 0.05)):
    image_list = list(root.glob('*/_start*/0*.jpg'))
    image_list.sort()
    split_point = [int(len(image_list) * split_ratio[0]), int(len(image_list) * (split_ratio[0] + split_ratio[1]))]
    return image_list[:split_point[0]], image_list[split_point[0]:split_point[1]], image_list[split_point[1]:]


def get_test_color_img(img_file_name, start_h, end_h, start_w, end_w, downsampling_factor, is_hsv):
    img = cv2.imread(img_file_name)
    downsampled_img = cv2.resize(img, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
    downsampled_img = downsampled_img[start_h:end_h, start_w:end_w, :]
    if is_hsv:
        downsampled_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2HSV_FULL)
    downsampled_img = np.array(downsampled_img, dtype="float32")
    return downsampled_img


def get_parent_folder_names(root, which_bag):
    training_folder_list = []
    rest_folder_list = []
    for i in range(1, 7):
        if i != which_bag:
            training_folder_list += list(root.glob('*' + str(i) + '/_start*/'))
        else:
            rest_folder_list = list(root.glob('*' + str(i) + '/_start*/'))

    training_folder_list.sort()
    rest_folder_list.sort()
    return training_folder_list, rest_folder_list


def downsample_and_crop_mask(mask, downsampling_factor, divide, suggested_h=None, suggested_w=None):
    downsampled_mask = cv2.resize(mask, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
    end_h_index = downsampled_mask.shape[0]
    end_w_index = downsampled_mask.shape[1]
    # divide is related to the pooling times of the teacher model
    indexes = np.where(downsampled_mask == 255)
    h = indexes[0].max() - indexes[0].min()
    w = indexes[1].max() - indexes[1].min()

    remainder_h = h % divide
    remainder_w = w % divide

    increment_h = divide - remainder_h
    increment_w = divide - remainder_w

    target_h = h + increment_h
    target_w = w + increment_w

    start_h = max(indexes[0].min() - increment_h // 2, 0)
    end_h = start_h + target_h

    start_w = max(indexes[1].min() - increment_w // 2, 0)
    end_w = start_w + target_w

    if suggested_h is not None:
        if suggested_h != h:
            remain_h = suggested_h - target_h
            start_h = max(start_h - remain_h // 2, 0)
            end_h = min(suggested_h + start_h, end_h_index)
            start_h = end_h - suggested_h

    if suggested_w is not None:
        if suggested_w != w:
            remain_w = suggested_w - target_w
            start_w = max(start_w - remain_w // 2, 0)
            end_w = min(suggested_w + start_w, end_w_index)
            start_w = end_w - suggested_w

    kernel = np.ones((5, 5), np.uint8)
    downsampled_mask_erode = cv2.erode(downsampled_mask, kernel, iterations=1)
    cropped_mask = downsampled_mask_erode[start_h:end_h, start_w:end_w]
    return cropped_mask, start_h, end_h, start_w, end_w


def read_selected_indexes(prefix_seq):
    selected_indexes = []
    with open(prefix_seq + 'selected_indexes') as fp:
        for line in fp:
            selected_indexes.append(int(line))

    stride = selected_indexes[1] - selected_indexes[0]
    return stride, selected_indexes


def read_visible_view_indexes(prefix_seq):
    visible_view_indexes = []
    with open(prefix_seq + 'visible_view_indexes_filtered') as fp:
        for line in fp:
            visible_view_indexes.append(int(line))

    visible_view_indexes_old = []
    with open(prefix_seq + 'visible_view_indexes') as fp:
        for line in fp:
            visible_view_indexes_old.append(int(line))

    if len(visible_view_indexes) != len(visible_view_indexes_old):
        print("We need to handle poses NOW!")
        exit(1)

    return visible_view_indexes


def read_camera_intrinsic_per_view(prefix_seq):
    camera_intrinsics = []
    param_count = 0
    temp_camera_intrincis = np.zeros((3, 4))
    with open(prefix_seq + 'camera_intrinsics_per_view') as fp:
        for line in fp:
            # Focal length
            if param_count == 0:
                temp_camera_intrincis[0][0] = float(line)
                temp_camera_intrincis[1][1] = float(line)
                param_count = 1
            elif param_count == 1:
                temp_camera_intrincis[0][2] = float(line)
                param_count = 2
            elif param_count == 2:
                temp_camera_intrincis[1][2] = float(line)
                temp_camera_intrincis[2][2] = 1.0
                camera_intrinsics.append(temp_camera_intrincis)
                temp_camera_intrincis = np.zeros((3, 4))
                param_count = 0
    return camera_intrinsics


def modify_camera_intrinsic_matrix(intrinsic_matrix, start_h, start_w, downsampling_factor):
    intrinsic_matrix_modified = np.copy(intrinsic_matrix)
    intrinsic_matrix_modified[0][0] = intrinsic_matrix[0][0] / downsampling_factor
    intrinsic_matrix_modified[1][1] = intrinsic_matrix[1][1] / downsampling_factor
    intrinsic_matrix_modified[0][2] = intrinsic_matrix[0][2] / downsampling_factor - start_w
    intrinsic_matrix_modified[1][2] = intrinsic_matrix[1][2] / downsampling_factor - start_h
    return intrinsic_matrix_modified


def read_point_cloud(prefix_seq):
    lists_3D_points = []
    plydata = PlyData.read(prefix_seq + "structure_filtered.ply")
    for n in range(plydata['vertex'].count):
        temp = list(plydata['vertex'][n])
        temp[0] = temp[0]
        temp[1] = temp[1]
        temp[2] = temp[2]
        temp.append(1.0)
        lists_3D_points.append(temp)
    return lists_3D_points


def read_view_indexes_per_point(prefix_seq, visible_view_indexes, point_cloud_count):
    # Read the view indexes per point into a 2-dimension binary matrix
    view_indexes_per_point = np.zeros((point_cloud_count, len(visible_view_indexes)))
    point_count = -1
    with open(prefix_seq + 'view_indexes_per_point_filtered') as fp:
        for line in fp:
            if int(line) < 0:
                point_count = point_count + 1
            else:
                view_indexes_per_point[point_count][visible_view_indexes.index(int(line))] = 1
    return view_indexes_per_point


def read_pose_data(prefix_seq):
    stream = open(prefix_seq + "motion.yaml", 'r')
    doc = yaml.load(stream)
    keys, values = doc.items()
    poses = values[1]
    return poses


def get_data_balancing_scale(poses, visible_view_count):
    traveling_distance = 0.0
    translation = np.zeros((3,), dtype=np.float)
    for i in range(visible_view_count):
        pre_translation = np.copy(translation)
        translation[0] = poses["poses[" + str(i) + "]"]['position']['x']
        translation[1] = poses["poses[" + str(i) + "]"]['position']['y']
        translation[2] = poses["poses[" + str(i) + "]"]['position']['z']

        if i >= 1:
            traveling_distance += np.linalg.norm(translation - pre_translation)
    traveling_distance /= visible_view_count
    return traveling_distance


def get_extrinsic_matrix_and_projection_matrix(poses, intrinsic_matrix, visible_view_count):
    projection_matrices = []
    extrinsic_matrices = []
    for i in range(visible_view_count):
        rigid_transform = quaternion_matrix(
            [poses["poses[" + str(i) + "]"]['orientation']['w'], poses["poses[" + str(i) + "]"]['orientation']['x'],
             poses["poses[" + str(i) + "]"]['orientation']['y'],
             poses["poses[" + str(i) + "]"]['orientation']['z']])
        rigid_transform[0][3] = poses["poses[" + str(i) + "]"]['position']['x']
        rigid_transform[1][3] = poses["poses[" + str(i) + "]"]['position']['y']
        rigid_transform[2][3] = poses["poses[" + str(i) + "]"]['position']['z']

        transform = np.asmatrix(rigid_transform)
        transform = np.linalg.inv(transform)

        extrinsic_matrices.append(transform)
        projection_matrices.append(np.dot(intrinsic_matrix, transform))

    return extrinsic_matrices, projection_matrices


def get_color_imgs(prefix_seq, visible_view_indexes, start_h, end_h, start_w, end_w, downsampling_factor, is_hsv=False):
    imgs = []
    for i in visible_view_indexes:
        img = cv2.imread((prefix_seq + "%08d.jpg") % (i))
        downsampled_img = cv2.resize(img, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
        cropped_downsampled_img = downsampled_img[start_h:end_h, start_w:end_w, :]
        if is_hsv:
            cropped_downsampled_img = cv2.cvtColor(cropped_downsampled_img, cv2.COLOR_BGR2HSV_FULL)
        imgs.append(cropped_downsampled_img)
    height, width, channel = imgs[0].shape
    imgs = np.array(imgs, dtype="float32")
    imgs = np.reshape(imgs, (-1, height, width, channel))
    return imgs


def get_contaminated_point_list(imgs, point_cloud, mask_boundary, inlier_percentage, projection_matrices,
                                extrinsic_matrices, is_hsv):
    contaminated_point_cloud_indexes = []
    if 0.0 < inlier_percentage < 1.0:
        point_cloud_contamination_accumulator = np.zeros(len(point_cloud))
        point_cloud_appearance_count = np.zeros(len(point_cloud))
        height, width, channel = imgs[0].shape
        sanity_array = []
        valid_frame_count = 0
        for i in range(len(projection_matrices)):
            img = imgs[i]
            projection_matrix = projection_matrices[i]
            extrinsic_matrix = extrinsic_matrices[i]

            img = np.array(img, dtype=np.float32)
            img = img / 255.0

            # imgs might be in HSV or BGR colorspace depending on the settings beyond this function
            if not is_hsv:
                img_filtered = cv2.bilateralFilter(src=img, d=7, sigmaColor=25, sigmaSpace=25)
                img_hsv = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2HSV_FULL)
            else:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR_FULL)
                img_filtered = cv2.bilateralFilter(src=img_bgr, d=7, sigmaColor=25, sigmaSpace=25)
                img_hsv = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2HSV_FULL)

            for j in range(len(point_cloud)):
                point_3d_position = np.asarray(point_cloud[j])
                point_3d_position_camera = np.asarray(extrinsic_matrix).dot(point_3d_position)
                point_3d_position_camera = point_3d_position_camera / point_3d_position_camera[3]
                point_3d_position_camera = np.reshape(point_3d_position_camera[:3], (3,))

                point_projected_undistorted = np.asarray(projection_matrix).dot(point_3d_position)
                point_projected_undistorted = point_projected_undistorted / point_projected_undistorted[2]

                if np.isnan(point_projected_undistorted[0]) or np.isnan(point_projected_undistorted[1]):
                    continue

                round_u = int(round(point_projected_undistorted[0]))
                round_v = int(round(point_projected_undistorted[1]))

                # We will treat this point as valid if it is projected onto the mask region
                if 0 <= round_u < width and 0 <= round_v < height:
                    if mask_boundary[round_v, round_u] > 220 and point_3d_position_camera[2] > 0.0:
                        point_to_camera_distance_2 = np.dot(point_3d_position_camera[:3], point_3d_position_camera[:3])
                        sanity_array.append(point_to_camera_distance_2 * img_hsv[round_v, round_u, 2])

            if len(sanity_array) != 0:
                valid_frame_count += 1
                # Use histogram to cluster into different contaminated levels
                hist, bin_edges = np.histogram(sanity_array, bins=np.arange(1000) * np.max(sanity_array) / 1000.0,
                                               density=True)
                histogram_percentage = hist * np.diff(bin_edges)
                percentage = inlier_percentage
                # Let's assume there are a certain percent of points in each frame that are not contaminated
                # Get sanity threshold from counting histogram bins
                max_index = np.argmax(histogram_percentage)
                histogram_sum = histogram_percentage[max_index]
                pos_counter = 1
                neg_counter = 1

                # Assume the sanity value is a one-peak distribution
                while True:
                    if max_index + pos_counter < len(histogram_percentage):
                        histogram_sum = histogram_sum + histogram_percentage[max_index + pos_counter]
                        pos_counter = pos_counter + 1
                        if histogram_sum >= percentage:
                            sanity_threshold_max = bin_edges[max_index + pos_counter]
                            sanity_threshold_min = bin_edges[max_index - neg_counter + 1]
                            break

                    if max_index - neg_counter >= 0:
                        histogram_sum = histogram_sum + histogram_percentage[max_index - neg_counter]
                        neg_counter = neg_counter + 1
                        if histogram_sum >= percentage:
                            sanity_threshold_max = bin_edges[max_index + pos_counter]
                            sanity_threshold_min = bin_edges[max_index - neg_counter + 1]
                            break

                for j in range(len(point_cloud)):
                    point_3d_position = np.asarray(point_cloud[j])
                    point_3d_position_camera = np.asarray(extrinsic_matrix).dot(point_3d_position)
                    point_3d_position_camera = point_3d_position_camera / point_3d_position_camera[3]
                    point_3d_position_camera = np.reshape(point_3d_position_camera[:3], (3,))

                    point_projected_undistorted = np.asarray(projection_matrix).dot(point_3d_position)
                    point_projected_undistorted = point_projected_undistorted / point_projected_undistorted[2]

                    if np.isnan(point_projected_undistorted[0]) or np.isnan(point_projected_undistorted[1]):
                        continue

                    round_u = int(round(point_projected_undistorted[0]))
                    round_v = int(round(point_projected_undistorted[1]))

                    if 0 <= round_u < width and 0 <= round_v < height:
                        if mask_boundary[round_v, round_u] > 220 and point_3d_position_camera[2] > 0.0:
                            point_to_camera_distance_2 = np.dot(point_3d_position_camera[:3],
                                                                point_3d_position_camera[:3])
                            sanity_value = point_to_camera_distance_2 * img_hsv[round_v, round_u, 2]
                            point_cloud_appearance_count[j] += 1
                            if sanity_value <= sanity_threshold_min or sanity_value >= sanity_threshold_max:
                                point_cloud_contamination_accumulator[j] += 1

        for i in range(point_cloud_contamination_accumulator.shape[0]):
            if point_cloud_contamination_accumulator[i] >= point_cloud_appearance_count[i] // 2:
                contaminated_point_cloud_indexes.append(i)

    print("{:d} points eliminated".format(len(contaminated_point_cloud_indexes)))
    return contaminated_point_cloud_indexes


def get_visible_count_per_point(view_indexes_per_point):
    appearing_count = np.reshape(np.sum(view_indexes_per_point, axis=-1), (-1, 1))
    return appearing_count


def generating_pos_and_increment(idx, visible_view_indexes, adjacent_range):
    # We use the remainder of the overall idx to retrieve the visible view
    visible_view_idx = idx % len(visible_view_indexes)

    adjacent_range_list = []
    adjacent_range_list.append(adjacent_range[0])
    adjacent_range_list.append(adjacent_range[1])

    if len(visible_view_indexes) <= 2 * adjacent_range_list[0]:
        adjacent_range_list[0] = len(visible_view_indexes) // 2

    if visible_view_idx <= adjacent_range_list[0] - 1:
        increment = random.randint(adjacent_range_list[0],
                                   min(adjacent_range_list[1], len(visible_view_indexes) - 1 - visible_view_idx))
    elif visible_view_idx >= len(visible_view_indexes) - adjacent_range_list[0]:
        increment = -random.randint(adjacent_range_list[0], min(adjacent_range_list[1], visible_view_idx))

    else:
        # which direction should we increment
        direction = random.randint(0, 1)
        if direction == 1:
            increment = random.randint(adjacent_range_list[0],
                                       min(adjacent_range_list[1], len(visible_view_indexes) - 1 - visible_view_idx))
        else:
            increment = -random.randint(adjacent_range_list[0], min(adjacent_range_list[1], visible_view_idx))

    return [visible_view_idx, increment]


def get_pair_color_imgs(prefix_seq, pair_indexes, start_h, end_h, start_w, end_w, downsampling_factor, is_hsv):
    imgs = []
    for i in pair_indexes:
        img = cv2.imread((prefix_seq + "%08d.jpg") % i)
        downsampled_img = cv2.resize(img, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
        downsampled_img = downsampled_img[start_h:end_h, start_w:end_w, :]
        if is_hsv:
            downsampled_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2HSV_FULL)
        imgs.append(downsampled_img)
    height, width, channel = imgs[0].shape
    imgs = np.array(imgs, dtype="float32")
    imgs = np.reshape(imgs, (-1, height, width, channel))
    return imgs


def get_torch_training_data(pair_images, pair_extrinsics, pair_projections, pair_indexes, point_cloud, mask_boundary,
                            view_indexes_per_point, contamination_point_list, appearing_count_per_point,
                            visible_view_indexes, use_view_indexes_per_point=False, visualize=False):
    height = pair_images.shape[1]
    width = pair_images.shape[2]
    pair_mask_imgs = []
    pair_sparse_depth_imgs = []

    count_weight = 5.0

    pair_flow_images = []
    flow_image_1 = np.zeros((height, width, 2), dtype=np.float32)
    flow_image_2 = np.zeros((height, width, 2), dtype=np.float32)

    pair_flow_mask_images = []
    flow_mask_image_1 = np.zeros((height, width, 1), dtype=np.float32)
    flow_mask_image_2 = np.zeros((height, width, 1), dtype=np.float32)

    point_projection_positions_1 = np.zeros((len(point_cloud) - len(contamination_point_list), 2), dtype=np.float32)
    point_projection_positions_2 = np.zeros((len(point_cloud) - len(contamination_point_list), 2), dtype=np.float32)

    # Calculate optical flows for each feature point
    for i in range(2):
        img = pair_images[i]
        if visualize:
            display_img = np.copy(img)
        projection_matrix = pair_projections[i]
        count = 0
        for j in range(len(point_cloud)):
            if j in contamination_point_list:
                continue
            point_3d_position = np.asarray(point_cloud[j])
            point_projected_undistorted = np.asarray(projection_matrix).dot(point_3d_position)
            point_projected_undistorted = point_projected_undistorted / point_projected_undistorted[2]

            if np.isnan(point_projected_undistorted[0]) or np.isnan(point_projected_undistorted[1]):
                continue

            round_u = int(round(point_projected_undistorted[0]))
            round_v = int(round(point_projected_undistorted[1]))

            if i == 0:
                point_projection_positions_1[count][0] = round_u
                point_projection_positions_1[count][1] = round_v

            elif i == 1:
                point_projection_positions_2[count][0] = round_u
                point_projection_positions_2[count][1] = round_v

            count += 1

    count = 0

    for i in range(len(point_cloud)):
        if i in contamination_point_list:
            continue
        u = point_projection_positions_1[count][0]
        v = point_projection_positions_1[count][1]
        u2 = point_projection_positions_2[count][0]
        v2 = point_projection_positions_2[count][1]

        if 0 <= u < width and 0 <= v < height:
            if mask_boundary[int(v), int(u)] > 220:
                distance = np.abs(float(u2 - u) / width) + np.abs(float(v2 - v) / height)
                if distance <= 1.0:
                    flow_image_1[int(v)][int(u)][0] = float(u2 - u) / width
                    flow_image_1[int(v)][int(u)][1] = float(v2 - v) / height

                    if use_view_indexes_per_point:
                        if view_indexes_per_point[i][visible_view_indexes.index(pair_indexes[0])] > 0.5:
                            flow_mask_image_1[int(v)][int(u)] = 1.0 - np.exp(
                                -appearing_count_per_point[i, 0] /
                                count_weight)
                        else:
                            flow_mask_image_1[int(v)][int(u)] = 0.0
                    else:
                        flow_mask_image_1[int(v)][int(u)] = 1.0 - np.exp(
                            -appearing_count_per_point[i, 0] /
                            count_weight)
                    # np.exp(-1.0 / (flow_factor * mean_flow_length) * distance)

        if 0 <= u2 < width and 0 <= v2 < height:
            if mask_boundary[int(v2), int(u2)] > 220:
                distance = np.abs(float(u - u2) / width) + np.abs(float(v - v2) / height)
                if distance <= 1.0:
                    flow_image_2[int(v2)][int(u2)][0] = float(u - u2) / width
                    flow_image_2[int(v2)][int(u2)][1] = float(v - v2) / height
                    if use_view_indexes_per_point:
                        if view_indexes_per_point[i][visible_view_indexes.index(pair_indexes[1])] > 0.5:
                            flow_mask_image_2[int(v2)][int(u2)] = 1.0 - np.exp(
                                -appearing_count_per_point[i, 0] /
                                count_weight)
                        else:
                            flow_mask_image_2[int(v2)][int(u2)] = 0.0
                    else:
                        flow_mask_image_2[int(v2)][int(u2)] = 1.0 - np.exp(
                            -appearing_count_per_point[i, 0] /
                            count_weight)
        count += 1
    for i in range(2):
        img = pair_images[i]

        if visualize:
            display_img = np.copy(img)

        projection_matrix = pair_projections[i]
        extrinsic_matrix = pair_extrinsics[i]

        masked_depth_img = np.zeros((height, width))
        mask_img = np.zeros((height, width))

        if use_view_indexes_per_point:
            for j in range(len(point_cloud)):
                if j in contamination_point_list:
                    continue
                point_3d_position = np.asarray(point_cloud[j])
                point_3d_position_camera = np.asarray(extrinsic_matrix).dot(point_3d_position)
                point_3d_position_camera = np.copy(point_3d_position_camera / point_3d_position_camera[3])

                point_projected_undistorted = np.asarray(projection_matrix).dot(point_3d_position)
                point_projected_undistorted = point_projected_undistorted / point_projected_undistorted[2]

                if np.isnan(point_projected_undistorted[0]) or np.isnan(point_projected_undistorted[1]):
                    continue

                round_u = int(round(point_projected_undistorted[0]))
                round_v = int(round(point_projected_undistorted[1]))
                if view_indexes_per_point[j][visible_view_indexes.index(pair_indexes[i])] > 0.5:
                    if 0 <= round_u < width and 0 <= round_v < height:
                        if mask_boundary[round_v, round_u] > 220 and point_3d_position_camera[2] > 0.0:
                            mask_img[round_v][
                                round_u] = 1.0 - np.exp(-appearing_count_per_point[j, 0] / count_weight)
                            masked_depth_img[round_v][round_u] = point_3d_position_camera[2]
                            if visualize:
                                cv2.circle(display_img, (round_u, round_v), 1,
                                           (0, int(mask_img[round_v][round_u] * 255), 0))
        else:
            for j in range(len(point_cloud)):
                if j in contamination_point_list:
                    continue
                point_3d_position = np.asarray(point_cloud[j])
                point_3d_position_camera = np.asarray(extrinsic_matrix).dot(point_3d_position)
                point_3d_position_camera = np.copy(point_3d_position_camera / point_3d_position_camera[3])

                point_projected_undistorted = np.asarray(projection_matrix).dot(point_3d_position)
                point_projected_undistorted[0] = point_projected_undistorted[0] / point_projected_undistorted[2]
                point_projected_undistorted[1] = point_projected_undistorted[1] / point_projected_undistorted[2]

                if np.isnan(point_projected_undistorted[0]) or np.isnan(point_projected_undistorted[1]):
                    continue

                round_u = int(round(point_projected_undistorted[0]))
                round_v = int(round(point_projected_undistorted[1]))
                if 0 <= round_u < width and 0 <= round_v < height:
                    if mask_boundary[round_v, round_u] > 220 and point_3d_position_camera[2] > 0.0:
                        # TODO: Binary mask only for depth scaling
                        mask_img[round_v][round_u] = 1.0 - np.exp(-appearing_count_per_point[j, 0] / count_weight)
                        masked_depth_img[round_v][round_u] = point_3d_position_camera[2]
                        if visualize:
                            cv2.circle(display_img, (round_u, round_v), 1,
                                       (0, int(mask_img[round_v][round_u] * 255), 0))
        if visualize:
            cv2.imshow("img", np.uint8(display_img))
            cv2.waitKey()

        pair_mask_imgs.append(mask_img)
        pair_sparse_depth_imgs.append(masked_depth_img)

    if visualize:
        cv2.destroyAllWindows()

    pair_flow_images.append(flow_image_1)
    pair_flow_images.append(flow_image_2)
    pair_flow_images = np.array(pair_flow_images, dtype="float32")
    pair_flow_images = np.reshape(pair_flow_images, (-1, height, width, 2))

    pair_flow_mask_images.append(flow_mask_image_1)
    pair_flow_mask_images.append(flow_mask_image_2)
    pair_flow_mask_images = np.array(pair_flow_mask_images, dtype="float32")
    pair_flow_mask_images = np.reshape(pair_flow_mask_images, (-1, height, width, 1))

    pair_mask_imgs = np.array(pair_mask_imgs, dtype="float32")
    pair_sparse_depth_imgs = np.array(pair_sparse_depth_imgs, dtype="float32")
    pair_mask_imgs = np.reshape(pair_mask_imgs, (-1, height, width, 1))
    pair_sparse_depth_imgs = np.reshape(pair_sparse_depth_imgs, (-1, height, width, 1))

    return [pair_mask_imgs, pair_sparse_depth_imgs, pair_flow_mask_images, pair_flow_images]


def init_fn(worker_id):
    np.random.seed(10086 + worker_id)


def init_net(net, type="kaiming", mode="fan_in", activation_mode="relu", distribution="normal"):
    assert (torch.cuda.is_available())
    net = net.cuda()
    if type == "glorot":
        glorot_weight_zero_bias(net, distribution=distribution)
    else:
        kaiming_weight_zero_bias(net, mode=mode, activation_mode=activation_mode, distribution=distribution)
    return net


def glorot_weight_zero_bias(model, distribution="uniform"):
    """
    Initalize parameters of all modules
    by initializing weights with glorot  uniform/xavier initialization,
    and setting biases to zero.
    Weights from batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    distribution: string
    """
    for module in model.modules():
        if hasattr(module, 'weight'):
            if not ('BatchNorm' in module.__class__.__name__):
                if distribution == "uniform":
                    torch.nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    torch.nn.init.xavier_normal_(module.weight, gain=1)
            else:
                torch.nn.init.constant_(module.weight, 1)
        if hasattr(module, 'bias'):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)


def kaiming_weight_zero_bias(model, mode="fan_in", activation_mode="relu", distribution="uniform"):
    if activation_mode == "leaky_relu":
        print("Leaky relu is not supported yet")
        assert False

    for module in model.modules():
        if hasattr(module, 'weight'):
            if not ('BatchNorm' in module.__class__.__name__):
                if distribution == "uniform":
                    torch.nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=activation_mode)
                else:
                    torch.nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=activation_mode)
            else:
                torch.nn.init.constant_(module.weight, 1)
        if hasattr(module, 'bias'):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)


def save_model(model, optimizer, epoch, step, model_path, failure_sequences, validation_loss):
    try:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'failure': failure_sequences,
            'validation': validation_loss
        }, str(model_path))
    except IOError:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'validation': validation_loss
        }, str(model_path))

    return


def visualize_color_image(title, images, rebias=False, is_hsv=False, idx=None):
    if idx is None:
        for i in range(images.shape[0]):
            image = images.data.cpu().numpy()[i]
            image = np.moveaxis(image, source=[0, 1, 2], destination=[2, 0, 1])
            if rebias:
                image = image * 0.5 + 0.5
            if is_hsv:
                image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR_FULL)
            cv2.imshow(title + "_" + str(i), image)
    else:
        for id in idx:
            image = images.data.cpu().numpy()[id]
            image = np.moveaxis(image, source=[0, 1, 2], destination=[2, 0, 1])
            if rebias:
                image = image * 0.5 + 0.5
            if is_hsv:
                image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR_FULL)
            cv2.imshow(title + "_" + str(id), image)


def visualize_depth_map(title, depth_maps, min_value_=None, max_value_=None, idx=None, color_mode=cv2.COLORMAP_HOT):
    min_value_list = []
    max_value_list = []
    if idx is None:
        for i in range(depth_maps.shape[0]):
            depth_map_cpu = depth_maps[i].data.cpu().numpy()

            if min_value_ is None and max_value_ is None:
                min_value = np.min(depth_map_cpu)
                max_value = np.max(depth_map_cpu)
                min_value_list.append(min_value)
                max_value_list.append(max_value)
            else:
                min_value = min_value_[i]
                max_value = max_value_[i]

            depth_map_cpu = np.moveaxis(depth_map_cpu, source=[0, 1, 2], destination=[2, 0, 1])
            depth_map_visualize = np.abs((depth_map_cpu - min_value) / (max_value - min_value) * 255)
            depth_map_visualize[depth_map_visualize > 255] = 255
            depth_map_visualize[depth_map_visualize <= 0.0] = 0
            depth_map_visualize = cv2.applyColorMap(np.uint8(depth_map_visualize), color_mode)
            cv2.imshow(title + "_" + str(i), depth_map_visualize)
        return min_value_list, max_value_list
    else:
        for id in idx:
            depth_map_cpu = depth_maps[id].data.cpu().numpy()

            if min_value_ is None and max_value_ is None:
                min_value = np.min(depth_map_cpu)
                max_value = np.max(depth_map_cpu)
                min_value_list.append(min_value)
                max_value_list.append(max_value)
            else:
                min_value = min_value_[id]
                max_value = max_value_[id]

            depth_map_cpu = np.moveaxis(depth_map_cpu, source=[0, 1, 2], destination=[2, 0, 1])
            depth_map_visualize = np.abs((depth_map_cpu - min_value) / (max_value - min_value) * 255)
            depth_map_visualize[depth_map_visualize > 255] = 255
            depth_map_visualize[depth_map_visualize <= 0.0] = 0
            depth_map_visualize = cv2.applyColorMap(np.uint8(depth_map_visualize), color_mode)
            cv2.imshow(title + "_" + str(id), depth_map_visualize)
        return min_value_list, max_value_list


def display_depth_map(depth_map, min_value=None, max_value=None, colormode=cv2.COLORMAP_JET):
    if min_value is None or max_value is None:
        min_value = np.min(depth_map)
        max_value = np.max(depth_map)
    depth_map_visualize = np.abs((depth_map - min_value) / (max_value - min_value) * 255)
    depth_map_visualize[depth_map_visualize > 255] = 255
    depth_map_visualize[depth_map_visualize <= 0.0] = 0
    depth_map_visualize = cv2.applyColorMap(np.uint8(depth_map_visualize), colormode)
    return depth_map_visualize


def draw_hsv(flows, title, idx=None):
    if idx is None:
        flows_cpu = flows.data.cpu().numpy()
        for i in range(flows_cpu.shape[0]):
            flow = np.moveaxis(flows_cpu[i], [0, 1, 2], [2, 0, 1])
            h, w = flow.shape[:2]
            fx, fy = flow[:, :, 0] * w, flow[:, :, 1] * h
            ang = np.arctan2(fy, fx) + np.pi
            v = np.sqrt(fx * fx + fy * fy)
            hsv = np.zeros((h, w, 3), np.uint8)
            hsv[..., 0] = ang * (180 / np.pi / 2)
            hsv[..., 1] = 255
            hsv[..., 2] = np.uint8(
                np.minimum(v, np.sqrt(0.01 * w * w + 0.01 * h * h)) / np.sqrt(0.01 * w * w + 0.01 * h * h) * 255)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow(title + str(i), bgr)
    else:
        flows_cpu = flows.data.cpu().numpy()
        for id in idx:
            flow = np.moveaxis(flows_cpu[id], [0, 1, 2], [2, 0, 1])
            h, w = flow.shape[:2]
            fx, fy = flow[:, :, 0] * w, flow[:, :, 1] * h
            ang = np.arctan2(fy, fx) + np.pi
            v = np.sqrt(fx * fx + fy * fy)
            hsv = np.zeros((h, w, 3), np.uint8)
            hsv[..., 0] = ang * (180 / np.pi / 2)
            hsv[..., 1] = 255
            hsv[..., 2] = np.uint8(
                np.minimum(v, np.sqrt(0.01 * w * w + 0.01 * h * h)) / np.sqrt(0.01 * w * w + 0.01 * h * h) * 255)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow(title + str(id), bgr)


def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.time().isoformat()
    log.write(unicode(json.dumps(data, sort_keys=True)))
    log.write(unicode('\n'))
    log.flush()


def point_cloud_from_depth(depth_map, color_img, mask_img, intrinsic_matrix, point_cloud_downsampling,
                           min_threshold=None, max_threshold=None):
    point_clouds = []
    height, width, channel = color_img.shape

    f_x = intrinsic_matrix[0, 0]
    c_x = intrinsic_matrix[0, 2]
    f_y = intrinsic_matrix[1, 1]
    c_y = intrinsic_matrix[1, 2]

    for h in range(height):
        for w in range(width):
            if h % point_cloud_downsampling == 0 and w % point_cloud_downsampling == 0 and mask_img[h, w] > 0.5:
                z = depth_map[h, w]
                x = (w - c_x) / f_x * z
                y = (h - c_y) / f_y * z
                r = color_img[h, w, 2]
                g = color_img[h, w, 1]
                b = color_img[h, w, 0]
                if max_threshold is not None and min_threshold is not None:
                    if np.max([r, g, b]) >= max_threshold and np.min([r, g, b]) <= min_threshold:
                        point_clouds.append((x, y, z, np.uint8(r), np.uint8(g), np.uint8(b)))
                else:
                    point_clouds.append((x, y, z, np.uint8(r), np.uint8(g), np.uint8(b)))

    point_clouds = np.array(point_clouds, dtype='float32')
    point_clouds = np.reshape(point_clouds, (-1, 6))
    return point_clouds


def write_point_cloud(path, point_cloud):
    point_clouds_list = []
    for i in range(point_cloud.shape[0]):
        point_clouds_list.append((point_cloud[i, 0], point_cloud[i, 1], point_cloud[i, 2], point_cloud[i, 3],
                                  point_cloud[i, 4], point_cloud[i, 5]))

    vertex = np.array(point_clouds_list,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(path)
    return


def display_training_output(idx, step, writer, colors_1, scaled_depth_maps_1):
    colors_display = vutils.make_grid(colors_1 * 0.5 + 0.5, normalize=False)
    colors_display_hsv = np.moveaxis(colors_display.data.cpu().numpy(),
                                     source=[0, 1, 2], destination=[2, 0, 1])
    colors_display_hsv[colors_display_hsv < 0.0] = 0.0
    colors_display_hsv[colors_display_hsv > 1.0] = 1.0
    colors_display_hsv = cv2.cvtColor(colors_display_hsv, cv2.COLOR_HSV2RGB_FULL)
    writer.add_image('Training/Images/Color_' + str(idx),
                     np.moveaxis(colors_display_hsv, source=[0, 1, 2], destination=[1, 2, 0]), step)

    depths_display = vutils.make_grid(scaled_depth_maps_1, normalize=True, scale_each=True)
    depths_display_hsv = cv2.applyColorMap(np.uint8(255 * np.moveaxis(depths_display.data.cpu().numpy(),
                                                                      source=[0, 1, 2], destination=[2, 0, 1])),
                                           cv2.COLORMAP_HOT)
    depths_display_hsv = cv2.cvtColor(depths_display_hsv, cv2.COLOR_BGR2RGB)
    writer.add_image('Training/Images/Depth_' + str(idx),
                     np.moveaxis(depths_display_hsv, source=[0, 1, 2], destination=[1, 2, 0]), step)

    return colors_display_hsv, depths_display_hsv


def display_depth_goal(idx, step, writer, goal_depth_map_1):
    depths_display = vutils.make_grid(goal_depth_map_1, normalize=True, scale_each=True)
    depths_display = cv2.applyColorMap(np.uint8(255 * np.moveaxis(depths_display.data.cpu().numpy(),
                                                                  source=[0, 1, 2], destination=[2, 0, 1])),
                                       cv2.COLORMAP_HOT)
    depths_display = cv2.cvtColor(depths_display, cv2.COLOR_BGR2RGB)
    writer.add_image('Training/Images/Goal_Depth_' + str(idx),
                     np.moveaxis(depths_display, source=[0, 1, 2], destination=[1, 2, 0]), step)
    return depths_display


def display_network_weights(depth_estimation_model_student, writer, step):
    for name, param in depth_estimation_model_student.named_parameters():
        writer.add_histogram("Weights/" + name, param.clone().cpu().data.numpy(), step)


def generate_training_output(colors_1, scaled_depth_maps_1, boundaries, intrinsic_matrices, is_hsv, epoch,
                             results_root):
    color_inputs_cpu = colors_1.data.cpu().numpy()
    pred_depths_cpu = scaled_depth_maps_1.data.cpu().numpy()
    boundaries_cpu = boundaries.data.cpu().numpy()
    intrinsics_cpu = intrinsic_matrices.data.cpu().numpy()
    color_imgs = []
    pred_depth_imgs = []

    for j in range(colors_1.shape[0]):
        color_img = color_inputs_cpu[j]
        pred_depth_img = pred_depths_cpu[j]

        color_img = np.moveaxis(color_img, source=[0, 1, 2], destination=[2, 0, 1])
        color_img = color_img * 0.5 + 0.5
        color_img[color_img < 0.0] = 0.0
        color_img[color_img > 1.0] = 1.0
        color_img = np.uint8(255 * color_img)
        if is_hsv:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_HSV2BGR_FULL)

        pred_depth_img = np.moveaxis(pred_depth_img, source=[0, 1, 2], destination=[2, 0, 1])

        if j == 0:
            # Write point cloud
            boundary = boundaries_cpu[j]
            intrinsic = intrinsics_cpu[j]
            boundary = np.moveaxis(boundary, source=[0, 1, 2], destination=[2, 0, 1])
            point_cloud = point_cloud_from_depth(pred_depth_img, color_img, boundary,
                                                 intrinsic,
                                                 point_cloud_downsampling=1)
            write_point_cloud(
                str(results_root / "point_cloud_epoch_{epoch}_index_{index}.ply".format(epoch=epoch,
                                                                                        index=j)),
                point_cloud)

        color_img = cv2.resize(color_img, dsize=(300, 300))
        pred_depth_img = cv2.resize(pred_depth_img, dsize=(300, 300))
        color_imgs.append(color_img)

        if j == 0:
            histr = cv2.calcHist([pred_depth_img], [0], None, histSize=[100], ranges=[0, 1000])
            plt.plot(histr, color='b')
            plt.xlim([0, 40])
            plt.savefig(
                str(results_root / 'generated_depth_hist_{epoch}.jpg'.format(epoch=epoch)))
            plt.clf()
        display_depth_img = display_depth_map(pred_depth_img)
        pred_depth_imgs.append(display_depth_img)

    final_color = color_imgs[0]
    final_pred_depth = pred_depth_imgs[0]
    for j in range(colors_1.shape[0] - 1):
        final_color = cv2.hconcat((final_color, color_imgs[j + 1]))
        final_pred_depth = cv2.hconcat((final_pred_depth, pred_depth_imgs[j + 1]))

    final = cv2.vconcat((final_color, final_pred_depth))
    cv2.imwrite(str(results_root / 'generated_mask_{epoch}.jpg'.format(epoch=epoch)),
                final)


def generate_validation_output(idx, step, writer, colors_1, scaled_depth_maps_1, boundaries, intrinsic_matrices, is_hsv,
                               results_root, which_bag):
    colors_display = vutils.make_grid(colors_1 * 0.5 + 0.5, normalize=False)
    colors_display_hsv = np.moveaxis(colors_display.data.cpu().numpy(),
                                     source=[0, 1, 2], destination=[2, 0, 1])
    colors_display_hsv[colors_display_hsv < 0.0] = 0.0
    colors_display_hsv[colors_display_hsv > 1.0] = 1.0
    colors_display_hsv = cv2.cvtColor(colors_display_hsv, cv2.COLOR_HSV2RGB_FULL)
    writer.add_image('Validation/Images/Color_' + str(idx),
                     np.moveaxis(colors_display_hsv, source=[0, 1, 2], destination=[1, 2, 0]), step)

    depths_display = vutils.make_grid(scaled_depth_maps_1, normalize=True, scale_each=True)
    depths_display_hsv = cv2.applyColorMap(np.uint8(255 * np.moveaxis(depths_display.data.cpu().numpy(),
                                                                      source=[0, 1, 2], destination=[2, 0, 1])),
                                           cv2.COLORMAP_HOT)
    depths_display_hsv = cv2.cvtColor(depths_display_hsv, cv2.COLOR_BGR2RGB)
    writer.add_image('Validation/Images/Depth_' + str(idx),
                     np.moveaxis(depths_display_hsv, source=[0, 1, 2], destination=[1, 2, 0]), step)

    color_inputs_cpu = colors_1.data.cpu().numpy()
    pred_depths_cpu = scaled_depth_maps_1.data.cpu().numpy()
    boundaries_cpu = boundaries.data.cpu().numpy()
    intrinsics_cpu = intrinsic_matrices.data.cpu().numpy()
    color_imgs = []
    pred_depth_imgs = []

    for j in range(colors_1.shape[0]):
        color_img = color_inputs_cpu[j]
        pred_depth_img = pred_depths_cpu[j]

        color_img = np.moveaxis(color_img, source=[0, 1, 2], destination=[2, 0, 1])
        color_img = color_img * 0.5 + 0.5
        color_img[color_img < 0.0] = 0.0
        color_img[color_img > 1.0] = 1.0
        color_img = np.uint8(255 * color_img)
        if is_hsv:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_HSV2BGR_FULL)

        pred_depth_img = np.moveaxis(pred_depth_img, source=[0, 1, 2], destination=[2, 0, 1])

        if j == 0:
            # Write point cloud
            boundary = boundaries_cpu[j]
            intrinsic = intrinsics_cpu[j]
            boundary = np.moveaxis(boundary, source=[0, 1, 2], destination=[2, 0, 1])
            point_cloud = point_cloud_from_depth(pred_depth_img, color_img, boundary,
                                                 intrinsic,
                                                 point_cloud_downsampling=1)
            write_point_cloud(
                str(results_root / "point_cloud_step_{step}_index_{index}_bag_{bag}.ply".format(step=step,
                                                                                                index=j,
                                                                                                bag=which_bag)),
                point_cloud)

        color_imgs.append(color_img)
        display_depth_img = display_depth_map(pred_depth_img)
        pred_depth_imgs.append(display_depth_img)

    final_color = color_imgs[0]
    final_pred_depth = pred_depth_imgs[0]
    for j in range(colors_1.shape[0] - 1):
        final_color = cv2.hconcat((final_color, color_imgs[j + 1]))
        final_pred_depth = cv2.hconcat((final_pred_depth, pred_depth_imgs[j + 1]))

    final = cv2.vconcat((final_color, final_pred_depth))
    cv2.imwrite(str(results_root / 'generated_mask_step_{step}_bag_{bag}.jpg'.format(step=step, bag=which_bag)),
                final)
    return


def generate_test_output(idx, step, writer, colors_1, scaled_depth_maps_1, boundaries, intrinsic_matrices, is_hsv,
                         results_root, which_bag, color_mode=cv2.COLORMAP_HOT):
    colors_display = vutils.make_grid(colors_1 * 0.5 + 0.5, normalize=False)
    colors_display_hsv = np.moveaxis(colors_display.data.cpu().numpy(),
                                     source=[0, 1, 2], destination=[2, 0, 1])
    colors_display_hsv[colors_display_hsv < 0.0] = 0.0
    colors_display_hsv[colors_display_hsv > 1.0] = 1.0
    colors_display_hsv = cv2.cvtColor(colors_display_hsv, cv2.COLOR_HSV2RGB_FULL)
    writer.add_image('Test/Images/Color_' + str(idx),
                     np.moveaxis(colors_display_hsv, source=[0, 1, 2], destination=[1, 2, 0]), step)

    depths_display = vutils.make_grid(scaled_depth_maps_1, normalize=True, scale_each=True)
    depths_display_hsv = cv2.applyColorMap(np.uint8(255 * np.moveaxis(depths_display.data.cpu().numpy(),
                                                                      source=[0, 1, 2], destination=[2, 0, 1])),
                                           colormap=color_mode)
    depths_display_hsv = cv2.cvtColor(depths_display_hsv, cv2.COLOR_BGR2RGB)
    writer.add_image('Test/Images/Depth_' + str(idx),
                     np.moveaxis(depths_display_hsv, source=[0, 1, 2], destination=[1, 2, 0]), step)

    color_inputs_cpu = colors_1.data.cpu().numpy()
    pred_depths_cpu = scaled_depth_maps_1.data.cpu().numpy()
    boundaries_cpu = boundaries.data.cpu().numpy()
    intrinsics_cpu = intrinsic_matrices.data.cpu().numpy()
    color_imgs = []
    pred_depth_imgs = []

    for j in range(colors_1.shape[0]):
        color_img = color_inputs_cpu[j]
        pred_depth_img = pred_depths_cpu[j]

        color_img = np.moveaxis(color_img, source=[0, 1, 2], destination=[2, 0, 1])
        color_img = color_img * 0.5 + 0.5
        color_img[color_img < 0.0] = 0.0
        color_img[color_img > 1.0] = 1.0
        color_img = np.uint8(255 * color_img)
        if is_hsv:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_HSV2BGR_FULL)

        pred_depth_img = np.moveaxis(pred_depth_img, source=[0, 1, 2], destination=[2, 0, 1])

        if j == 0:
            # Write point cloud
            boundary = boundaries_cpu[j]
            intrinsic = intrinsics_cpu[j]
            boundary = np.moveaxis(boundary, source=[0, 1, 2], destination=[2, 0, 1])
            point_cloud = point_cloud_from_depth(pred_depth_img, color_img, boundary,
                                                 intrinsic,
                                                 point_cloud_downsampling=1)
            write_point_cloud(
                str(results_root / "test_point_cloud_step_{step}_bag_{bag}.ply".format(step=step, bag=which_bag)),
                point_cloud)

        color_imgs.append(color_img)
        display_depth_img = display_depth_map(pred_depth_img, colormode=color_mode)
        pred_depth_imgs.append(display_depth_img)

    final_color = color_imgs[0]
    final_pred_depth = pred_depth_imgs[0]
    for j in range(colors_1.shape[0] - 1):
        final_color = cv2.hconcat((final_color, color_imgs[j + 1]))
        final_pred_depth = cv2.hconcat((final_pred_depth, pred_depth_imgs[j + 1]))

    final = cv2.vconcat((final_color, final_pred_depth))
    cv2.imwrite(str(results_root / 'generated_mask_step_{step}_bag_{bag}.jpg'.format(step=step, bag=which_bag)),
                final)
    return


def point_cloud_from_depth_and_initial_pose(depth_map, color_img, mask_img, intrinsic_matrix, translation, rotation,
                                            point_cloud_downsampling,
                                            min_threshold=None, max_threshold=None):
    point_clouds = []
    height, width, channel = color_img.shape

    f_x = intrinsic_matrix[0, 0]
    c_x = intrinsic_matrix[0, 2]
    f_y = intrinsic_matrix[1, 1]
    c_y = intrinsic_matrix[1, 2]

    z_min = -1
    z_max = -1

    for h in range(height):
        for w in range(width):
            if h % point_cloud_downsampling == 0 and w % point_cloud_downsampling == 0 and mask_img[h, w] > 0.5:
                z = depth_map[h, w]
                if z_min == -1:
                    z_min = z
                    z_max = z
                else:
                    z_min = min(z, z_min)
                    z_max = max(z, z_max)

    scale = 20.0 / (z_max - z_min)

    for h in range(height):
        for w in range(width):
            if h % point_cloud_downsampling == 0 and w % point_cloud_downsampling == 0 and mask_img[h, w] > 0.5:
                z = depth_map[h, w]
                x = (w - c_x) / f_x * z
                y = (h - c_y) / f_y * z
                position = np.array([x * scale, y * scale, z * scale], dtype=np.float32).reshape((3, 1))
                transformed_position = np.matmul(rotation, position) + translation.reshape((3, 1))

                r = color_img[h, w, 2]
                g = color_img[h, w, 1]
                b = color_img[h, w, 0]
                if max_threshold is not None and min_threshold is not None:
                    if np.max([r, g, b]) >= max_threshold and np.min([r, g, b]) <= min_threshold:
                        point_clouds.append((transformed_position[0], transformed_position[1], transformed_position[2],
                                             np.uint8(r), np.uint8(g), np.uint8(b)))
                else:
                    point_clouds.append((transformed_position[0], transformed_position[1], transformed_position[2],
                                         np.uint8(r), np.uint8(g), np.uint8(b)))

    point_clouds = np.array(point_clouds, dtype='float32')
    point_clouds = np.reshape(point_clouds, (-1, 6))
    return point_clouds


def read_pose_messages_from_tracker(file_path):
    translation_array = []
    rotation_array = []
    with open(file_path, "r") as filestream:
        for count, line in enumerate(filestream):
            # Skip the header
            if count == 0:
                continue
            array = line.split(",")
            array = array[5:]
            array = np.array(array, dtype=np.float64)
            translation_array.append(array[:3])
            qx, qy, qz, qw = array[3:]
            rotation_matrix = quaternion_matrix([qw, qx, qy, qz])
            rotation_array.append(rotation_matrix[:3, :3])
    return translation_array, rotation_array


def write_test_output_with_initial_pose(results_root, colors_1, scaled_depth_maps_1, boundaries, intrinsic_matrices,
                                        is_hsv,
                                        image_indexes,
                                        translation_dict, rotation_dict, color_mode=cv2.COLORMAP_HOT):
    color_inputs_cpu = colors_1.data.cpu().numpy()
    pred_depths_cpu = (boundaries * scaled_depth_maps_1).data.cpu().numpy()
    boundaries_cpu = boundaries.data.cpu().numpy()
    intrinsics_cpu = intrinsic_matrices.data.cpu().numpy()

    for j in range(colors_1.shape[0]):
        print("processing {}...".format(image_indexes[j]))
        color_img = color_inputs_cpu[j]
        pred_depth_img = pred_depths_cpu[j]

        color_img = np.moveaxis(color_img, source=[0, 1, 2], destination=[2, 0, 1])
        color_img = color_img * 0.5 + 0.5
        color_img[color_img < 0.0] = 0.0
        color_img[color_img > 1.0] = 1.0
        color_img = np.uint8(255 * color_img)
        if is_hsv:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_HSV2BGR_FULL)

        pred_depth_img = np.moveaxis(pred_depth_img, source=[0, 1, 2], destination=[2, 0, 1])

        # Write point cloud
        boundary = boundaries_cpu[j]
        intrinsic = intrinsics_cpu[j]
        boundary = np.moveaxis(boundary, source=[0, 1, 2], destination=[2, 0, 1])
        point_cloud = point_cloud_from_depth_and_initial_pose(pred_depth_img, color_img, boundary, intrinsic,
                                                              translation=translation_dict[image_indexes[j]],
                                                              rotation=rotation_dict[image_indexes[j]],
                                                              point_cloud_downsampling=1,
                                                              min_threshold=None, max_threshold=None)

        write_point_cloud(str(results_root / ("test_point_cloud_" + image_indexes[j] + ".ply")), point_cloud)
        cv2.imwrite(str(results_root / ("test_color_" + image_indexes[j] + ".jpg")), color_img)
        display_depth_img = display_depth_map(pred_depth_img, colormode=color_mode)
        cv2.imwrite(str(results_root / ("test_depth_" + image_indexes[j] + ".jpg")), display_depth_img)

    return


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < np.finfo(float).eps * 4.0:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])


def read_initial_pose_file(file_path):
    frame_index_array = []
    translation_dict = {}
    rotation_dict = {}

    with open(file_path, "r") as filestream:
        for line in filestream:
            array = line.split(", ")
            array = np.array(array, dtype=np.float64)
            frame_index_array.append(int(array[0]))
            translation_dict["{:08d}".format(int(array[0]))] = array[1:4]
            rotation_matrix = quaternion_matrix(array[4:])
            # flip y and z axes
            rotation_matrix[:3, 1] = -rotation_matrix[:3, 1]
            rotation_matrix[:3, 2] = -rotation_matrix[:3, 2]
            rotation_dict["{:08d}".format(int(array[0]))] = rotation_matrix[:3, :3]
    frame_index_array.sort()
    return frame_index_array, translation_dict, rotation_dict


def get_filenames_from_frame_indexes(bag_root, frame_index_array):
    test_image_list = []
    for index in frame_index_array:
        temp = list(bag_root.glob('*/*{:08d}.jpg'.format(index)))
        if len(temp) == 0:
            print index
        else:
            test_image_list.append(temp[0])
    test_image_list.sort()
    return test_image_list


def outlier_detection(i, epoch, sparse_flow_weight, sparse_flow_loss, display, flows_1, flows_from_depth_1,
                      flow_masks_1,
                      flows_2, flows_from_depth_2, flow_masks_2, folders, boundaries, scaled_depth_maps_1,
                      scaled_depth_maps_2, colors_1, colors_2, is_hsv):
    print("batch {:d} in epoch {:d} has large loss {:.5f}".format(i, epoch,
                                                                  sparse_flow_weight * sparse_flow_loss.item()))

    losses_1 = display(
        [flows_1, flows_from_depth_1, flow_masks_1])
    losses_2 = display(
        [flows_2, flows_from_depth_2, flow_masks_2])

    indice_1 = torch.argmax(losses_1, dim=0, keepdim=False)
    indice_2 = torch.argmax(losses_2, dim=0, keepdim=False)

    print("pair 1 max loss: {:.5f}, pair 2 max loss: {:.5f}".format(torch.max(losses_1).item(),
                                                                    torch.max(losses_2).item()))
    print(folders[indice_1.item()], folders[indice_2.item()])
    visualize_color_image("mask_sample_", boundaries, rebias=False, is_hsv=False,
                          idx=[indice_1.item(), indice_2.item()])

    visualize_color_image("original color_1_sample_", colors_1, rebias=True,
                          is_hsv=is_hsv, idx=[indice_1.item()])
    visualize_depth_map("depth_1_sample_", scaled_depth_maps_1, idx=[indice_1.item()])
    draw_hsv(flows_1, "sparse_flow_1_sample_", idx=[indice_1.item()])
    draw_hsv(flows_from_depth_1, "flow from depth_1_sample_", idx=[indice_1.item()])

    visualize_color_image("original color_2_sample_", colors_2, rebias=True,
                          is_hsv=is_hsv, idx=[indice_2.item()])
    visualize_depth_map("depth_2_sample_", scaled_depth_maps_2, idx=[indice_2.item()])
    draw_hsv(flows_2, "sparse_flow_2_sample_", idx=[indice_2.item()])
    draw_hsv(flows_from_depth_2, "flow from depth_2_sample_", idx=[indice_2.item()])
    cv2.waitKey()
    cv2.destroyAllWindows()


def outlier_detection_processing(failure_threshold, sparse_masked_l1_loss_detector, flows,
                                 flows_from_depth, flow_masks):
    failure_detection_loss = sparse_masked_l1_loss_detector(
        [flows, flows_from_depth, flow_masks])
    indexes = []
    for j in range(failure_detection_loss.shape[0]):
        if failure_detection_loss[j].item() > failure_threshold:
            indexes.append(j)
    return indexes, failure_detection_loss


def learn_from_teacher(boundaries, colors_1, colors_2, depth_estimation_model_teacher, depth_estimation_model_student,
                       scale_invariant_loss):
    # Predicted depth from teacher model (where sparse signal can be easily propagated)
    goal_depth_maps_1 = depth_estimation_model_teacher(colors_1)
    goal_depth_maps_2 = depth_estimation_model_teacher(colors_2)
    # Predicted depth from student model
    predicted_depth_maps_1 = depth_estimation_model_student(colors_1)
    predicted_depth_maps_2 = depth_estimation_model_student(colors_2)

    abs_goal_depth_maps_1 = torch.abs(goal_depth_maps_1)
    abs_goal_depth_maps_2 = torch.abs(goal_depth_maps_2)

    abs_predicted_depth_maps_1 = torch.abs(predicted_depth_maps_1)
    abs_predicted_depth_maps_2 = torch.abs(predicted_depth_maps_2)

    loss = 0.5 * scale_invariant_loss(
        [abs_predicted_depth_maps_1, abs_goal_depth_maps_1, boundaries]) + \
           0.5 * scale_invariant_loss(
        [abs_predicted_depth_maps_2, abs_goal_depth_maps_2, boundaries])
    return loss, torch.abs(predicted_depth_maps_1), torch.abs(predicted_depth_maps_2), \
           torch.abs(goal_depth_maps_1), torch.abs(goal_depth_maps_2)


def learn_from_sfm(colors_1, colors_2, sparse_depths_1, sparse_depths_2, sparse_depth_masks_1, sparse_depth_masks_2,
                   depth_estimation_model_student, depth_scaling_layer, sparse_flow_weight, flow_from_depth_layer, boundaries,
                   translations, rotations, intrinsic_matrices, translations_inverse, rotations_inverse,
                   flow_masks_1, flow_masks_2, flows_1, flows_2, enable_failure_detection,
                   sparse_masked_l1_loss, depth_consistency_weight, depth_warping_layer, masked_log_l2_loss,
                   batch, epoch, failure_threshold, sparse_masked_l1_loss_detector, epoch_failure_sequences,
                   folders, batch_size, visualize, scale_std_loss_weight):
    # Predicted depth from student model
    predicted_depth_maps_1 = depth_estimation_model_student(colors_1)
    predicted_depth_maps_2 = depth_estimation_model_student(colors_2)

    # print(torch.min(predicted_depth_maps_1), torch.min(predicted_depth_maps_2))
    scaled_depth_maps_1, normalized_scale_std_1 = depth_scaling_layer(
        [torch.abs(predicted_depth_maps_1), sparse_depths_1, sparse_depth_masks_1])
    scaled_depth_maps_2, normalized_scale_std_2 = depth_scaling_layer(
        [torch.abs(predicted_depth_maps_2), sparse_depths_2, sparse_depth_masks_2])

    scale_std_loss = 0.5 * normalized_scale_std_1 + 0.5 * normalized_scale_std_2
    flows_from_depth_1 = torch.tensor(0.0).float().cuda()
    flows_from_depth_2 = torch.tensor(0.0).float().cuda()
    depth_consistency_loss = torch.tensor(0.0).float().cuda()
    sparse_flow_loss = torch.tensor(0.0).float().cuda()
    warped_depth_maps_2_to_1 = 0.0
    warped_depth_maps_1_to_2 = 0.0

    if sparse_flow_weight > 0.0:
        # Sparse optical flow loss
        # Flow maps calculated using predicted dense depth maps and camera poses
        # should agree with the sparse flows of feature points from SfM
        flows_from_depth_1 = flow_from_depth_layer(
            [scaled_depth_maps_1, boundaries, translations, rotations,
             intrinsic_matrices])
        flows_from_depth_2 = flow_from_depth_layer(
            [scaled_depth_maps_2, boundaries, translations_inverse, rotations_inverse,
             intrinsic_matrices])
        flow_masks_1 = flow_masks_1 * boundaries
        flow_masks_2 = flow_masks_2 * boundaries
        flows_1 = flows_1 * boundaries
        flows_2 = flows_2 * boundaries
        flows_from_depth_1 = flows_from_depth_1 * boundaries
        flows_from_depth_2 = flows_from_depth_2 * boundaries
        # If we do not try to detect any failure case from SfM
        if not enable_failure_detection:
            sparse_flow_loss = 0.5 * sparse_masked_l1_loss(
                [flows_1, flows_from_depth_1, flow_masks_1]) + \
                              0.5 * sparse_masked_l1_loss(
                [flows_2, flows_from_depth_2, flow_masks_2])

    if depth_consistency_weight > 0.0:
        # Depth consistency loss
        warped_depth_maps_2_to_1, intersect_masks_1 = depth_warping_layer(
            [scaled_depth_maps_1, scaled_depth_maps_2, boundaries, translations, rotations,
             intrinsic_matrices])
        warped_depth_maps_1_to_2, intersect_masks_2 = depth_warping_layer(
            [scaled_depth_maps_2, scaled_depth_maps_1, boundaries, translations_inverse,
             rotations_inverse,
             intrinsic_matrices])
        depth_consistency_loss = 0.5 * masked_log_l2_loss(
            [scaled_depth_maps_1, warped_depth_maps_2_to_1, intersect_masks_1, translations]) + \
                                 0.5 * masked_log_l2_loss(
            [scaled_depth_maps_2, warped_depth_maps_1_to_2, intersect_masks_2, translations])
        if visualize:
            visualize_color_image("color_1_sample_", colors_1, rebias=True, is_hsv=True, idx=[0, ])
            visualize_color_image("color_2_sample_", colors_2, rebias=True, is_hsv=True, idx=[0, ])
            min_list, max_list = visualize_depth_map("depth_1_sample_", scaled_depth_maps_1 * boundaries, idx=[0, ],
                                                     color_mode=cv2.COLORMAP_JET)
            visualize_depth_map("depth_2_sample_", scaled_depth_maps_2 * boundaries, min_value_=min_list,
                                max_value_=max_list, idx=[0, ], color_mode=cv2.COLORMAP_JET)
            # visualize_depth_map("depth_2_sample_", scaled_depth_maps_2 * boundaries)
            visualize_depth_map("2_to_1_depth_sample_", intersect_masks_1 * warped_depth_maps_2_to_1,
                                min_value_=min_list, max_value_=max_list, idx=[0, ], color_mode=cv2.COLORMAP_JET)
            visualize_depth_map("1_to_2_depth_sample_", intersect_masks_2 * warped_depth_maps_1_to_2,
                                min_value_=min_list, max_value_=max_list, idx=[0, ], color_mode=cv2.COLORMAP_JET)
            draw_hsv(flows_from_depth_1, "flow1_sample_", idx=[0, ])
            draw_hsv(flows_from_depth_2, "flow2_sample_", idx=[0, ])
            cv2.waitKey()
    loss = 0.0
    # Bootstrapping data cleaning method
    if enable_failure_detection:
        failure_indexes_1, sparse_flow_losses_1 = outlier_detection_processing(failure_threshold,
                                                                              sparse_masked_l1_loss_detector,
                                                                              flows_1,
                                                                              flows_from_depth_1,
                                                                              flow_masks_1)
        failure_indexes_2, sparse_flow_losses_2 = outlier_detection_processing(failure_threshold,
                                                                              sparse_masked_l1_loss_detector,
                                                                              flows_2,
                                                                              flows_from_depth_2,
                                                                              flow_masks_2)
        for index in failure_indexes_1:
            epoch_failure_sequences[folders[index]] = 1
            sparse_flow_losses_1[index] = torch.tensor(0.0).float().cuda()

        for index in failure_indexes_2:
            epoch_failure_sequences[folders[index]] = 1
            sparse_flow_losses_2[index] = torch.tensor(0.0).float().cuda()

        if batch_size != len(failure_indexes_1) and batch_size != len(failure_indexes_2):
            sparse_flow_loss = torch.tensor(0.5).float().cuda() * (
                    torch.sum(sparse_flow_losses_1) / torch.tensor(
                batch_size - len(failure_indexes_1)).float().cuda() + torch.sum(
                sparse_flow_losses_2) / torch.tensor(
                batch_size - len(failure_indexes_2)).float().cuda())
        else:
            sparse_flow_loss = torch.tensor(0.5).float().cuda() * torch.mean(sparse_flow_losses_1) + \
                torch.tensor(0.5).float().cuda() * torch.mean(sparse_flow_losses_2)

        loss = depth_consistency_weight * depth_consistency_loss + sparse_flow_weight * sparse_flow_loss + scale_std_loss_weight * scale_std_loss
    else:
        loss = depth_consistency_weight * depth_consistency_loss + sparse_flow_weight * sparse_flow_loss + scale_std_loss_weight * scale_std_loss

    return loss, scaled_depth_maps_1, scaled_depth_maps_2, epoch_failure_sequences, \
           depth_consistency_loss, sparse_flow_loss, scale_std_loss, warped_depth_maps_2_to_1, warped_depth_maps_1_to_2, predicted_depth_maps_1, sparse_flow_losses_1, sparse_flow_losses_2


def save_student_model(model_root, depth_estimation_model_student, optimizer, epoch,
                       step, failure_sequences, model_path_student, validation_losses, best_validation_losses, save_best_only):
    model_path_epoch_student = model_root / 'checkpoint_student_model_epoch_{epoch}.pt'.format(epoch=epoch)
    validation_losses = np.array(validation_losses)
    best_validation_losses = np.array(best_validation_losses)

    # Checkpoint model
    save_model(model=depth_estimation_model_student, optimizer=optimizer,
               epoch=epoch + 1, step=step,
               model_path=model_path_epoch_student, failure_sequences=failure_sequences,
               validation_loss=validation_losses)

    # Best model
    # If we use the validation loss to select our model
    if save_best_only:
        # Save best validation loss model
        if calculate_outlier_robust_validation_loss(validation_losses, best_validation_losses) < 0.0:
            print("Found better model in terms of validation loss: {:.5f}".format(np.mean(validation_losses)))
            save_model(model=depth_estimation_model_student, optimizer=optimizer,
                       epoch=epoch + 1, step=step,
                       model_path=model_path_student, failure_sequences=failure_sequences,
                       validation_loss=validation_losses)
            return validation_losses
        else:
            return best_validation_losses

    else:
        save_model(model=depth_estimation_model_student, optimizer=optimizer,
                   epoch=epoch + 1, step=step,
                   model_path=model_path_student, failure_sequences=failure_sequences,
                   validation_loss=validation_losses)
        return validation_losses


def save_teacher_model(model_root, depth_estimation_model_teacher, optimizer, epoch,
                       step, failure_sequences, model_path_teacher, validation_losses, best_validation_losses, save_best_only):
    model_path_epoch_teacher = model_root / 'checkpoint_teacher_model_epoch_{epoch}.pt'.format(epoch=epoch)
    validation_losses = np.array(validation_losses)
    best_validation_losses = np.array(best_validation_losses)

    # Checkpoint model
    save_model(model=depth_estimation_model_teacher, optimizer=optimizer,
               epoch=epoch + 1, step=step,
               model_path=model_path_epoch_teacher, failure_sequences=failure_sequences,
               validation_loss=validation_losses)
    # Best model
    # If we use the validation loss to select our model
    if save_best_only:
        # Save best validation loss model
        if calculate_outlier_robust_validation_loss(validation_losses, best_validation_losses) < 0.0:
            print("Found better model in terms of validation loss: {:.5f}".format(np.mean(validation_losses)))
            save_model(model=depth_estimation_model_teacher, optimizer=optimizer,
                       epoch=epoch + 1, step=step,
                       model_path=model_path_teacher, failure_sequences=failure_sequences,
                       validation_loss=validation_losses)
            return validation_losses
        else:
            return best_validation_losses

    else:
        save_model(model=depth_estimation_model_teacher, optimizer=optimizer,
                   epoch=epoch + 1, step=step,
                   model_path=model_path_teacher, failure_sequences=failure_sequences,
                   validation_loss=validation_losses)
        return validation_losses


def network_validation(writer, validation_loader, batch_size, epoch, depth_estimation_model_student, device,
                       depth_scaling_layer,
                       sparse_flow_weight, flow_from_depth_layer, sparse_masked_l1_loss, depth_consistency_weight,
                       masked_log_l2_loss,
                       is_hsv, depth_warping_layer, results_root, tq, which_bag):
    # Validation
    # Variable initialization
    depth_consistency_loss = torch.tensor(0.0).float().cuda()
    sparse_flow_loss = torch.tensor(0.0).float().cuda()
    scale_std_loss = torch.tensor(0.0).float().cuda()
    validation_losses = []
    validation_sparse_flow_losses = []
    validation_depth_consistency_losses = []
    for param in depth_estimation_model_student.parameters():
        param.requires_grad = False

    sample_batch = np.random.randint(low=0, high=len(validation_loader))
    for batch, (
            colors_1, colors_2, sparse_depths_1, sparse_depths_2, sparse_depth_masks_1,
            sparse_depth_masks_2,
            flows_1,
            flows_2, flow_masks_1, flow_masks_2, boundaries, rotations,
            rotations_inverse, translations, translations_inverse, intrinsic_matrices,
            folders) in enumerate(
        validation_loader):

        colors_1, colors_2, \
        sparse_depths_1, sparse_depths_2, sparse_depth_masks_1, sparse_depth_masks_2, flows_1, flows_2, flow_masks_1, \
        flow_masks_2, \
        boundaries, rotations, rotations_inverse, translations, translations_inverse, intrinsic_matrices = \
            colors_1.to(device), colors_2.to(device), \
            sparse_depths_1.to(device), sparse_depths_2.to(device), \
            sparse_depth_masks_1.to(device), sparse_depth_masks_2.to(device), flows_1.to(
                device), flows_2.to(
                device), flow_masks_1.to(device), flow_masks_2.to(device), \
            boundaries.to(device), rotations.to(device), \
            rotations_inverse.to(device), translations.to(device), translations_inverse.to(
                device), intrinsic_matrices.to(device)

        # Binarize the boundaries
        boundaries = torch.where(boundaries >= torch.tensor(0.9).float().cuda(),
                                 torch.tensor(1.0).float().cuda(), torch.tensor(0.0).float().cuda())
        # Remove invalid regions of color images
        colors_1 = boundaries * colors_1
        colors_2 = boundaries * colors_2

        # Predicted depth from student model
        predicted_depth_maps_1 = depth_estimation_model_student(colors_1)
        predicted_depth_maps_2 = depth_estimation_model_student(colors_2)
        scaled_depth_maps_1, normalized_scale_std_1 = depth_scaling_layer(
            [torch.abs(predicted_depth_maps_1), sparse_depths_1, sparse_depth_masks_1])
        scaled_depth_maps_2, normalized_scale_std_2 = depth_scaling_layer(
            [torch.abs(predicted_depth_maps_2), sparse_depths_2, sparse_depth_masks_2])

        if sparse_flow_weight > 0.0:
            # Sparse optical flow loss
            # Optical flow maps calculated using predicted dense depth maps and camera poses
            # should agree with the sparse optical flows of feature points from SfM
            flows_from_depth_1 = flow_from_depth_layer(
                [scaled_depth_maps_1, boundaries, translations, rotations,
                 intrinsic_matrices])
            flows_from_depth_2 = flow_from_depth_layer(
                [scaled_depth_maps_2, boundaries, translations_inverse, rotations_inverse,
                 intrinsic_matrices])
            flow_masks_1 = flow_masks_1 * boundaries
            flow_masks_2 = flow_masks_2 * boundaries
            flows_1 = flows_1 * boundaries
            flows_2 = flows_2 * boundaries
            flows_from_depth_1 = flows_from_depth_1 * boundaries
            flows_from_depth_2 = flows_from_depth_2 * boundaries
            # If we do not try to detect any failure case from SfM
            sparse_flow_loss = 0.5 * sparse_masked_l1_loss(
                [flows_1, flows_from_depth_1, flow_masks_1]) + \
                              0.5 * sparse_masked_l1_loss(
                [flows_2, flows_from_depth_2, flow_masks_2])

        if depth_consistency_weight > 0.0:
            # Depth consistency loss
            warped_depth_maps_2_to_1, intersect_masks_1 = depth_warping_layer(
                [scaled_depth_maps_1, scaled_depth_maps_2, boundaries, translations, rotations,
                 intrinsic_matrices])
            warped_depth_maps_1_to_2, intersect_masks_2 = depth_warping_layer(
                [scaled_depth_maps_2, scaled_depth_maps_1, boundaries, translations_inverse,
                 rotations_inverse,
                 intrinsic_matrices])
            depth_consistency_loss = 0.5 * masked_log_l2_loss(
                [scaled_depth_maps_1, warped_depth_maps_2_to_1, intersect_masks_1, translations]) + \
                                     0.5 * masked_log_l2_loss(
                [scaled_depth_maps_2, warped_depth_maps_1_to_2, intersect_masks_2, translations])

        loss = depth_consistency_weight * depth_consistency_loss + sparse_flow_weight * sparse_flow_loss

        # Avoid the effects of nan samples
        if not np.isnan(loss.item()):
            validation_losses.append(loss.item())
            validation_sparse_flow_losses.append(sparse_flow_weight * sparse_flow_loss.item())
            validation_depth_consistency_losses.append(
                depth_consistency_weight * depth_consistency_loss.item())
            tq.set_postfix(loss='{:.5f} {:.5f}'.format(np.mean(validation_losses), loss.item()),
                           loss_depth_consistency='{:.5f} {:.5f}'.format(
                               np.mean(validation_depth_consistency_losses),
                               depth_consistency_weight * depth_consistency_loss.item()),
                           loss_sparse_flow='{:.5f} {:.5f}'.format(np.mean(validation_sparse_flow_losses),
                                                                  sparse_flow_weight * sparse_flow_loss.item()))
        tq.update(batch_size)

        if batch == sample_batch:
            generate_validation_output(1, epoch, writer, colors_1, scaled_depth_maps_1 * boundaries, boundaries,
                                       intrinsic_matrices,
                                       is_hsv, results_root, which_bag)

    # TensorboardX
    writer.add_scalars('Validation', {'overall': np.mean(validation_losses),
                                      'depth consistency': np.mean(validation_depth_consistency_losses),
                                      'sparse opt': np.mean(validation_sparse_flow_losses)}, epoch)

    return np.mean(validation_losses), validation_losses


def calculate_outlier_robust_validation_loss(validation_losses, previous_validation_losses):
    if len(validation_losses) == len(previous_validation_losses):
        differences = validation_losses - previous_validation_losses

        positive = np.sum(np.sum(np.int32(differences > 0.0)) * (differences > 0.0) * differences)
        negative = np.sum(np.sum(np.int32(differences < 0.0)) * (differences < 0.0) * differences)
        return positive + negative
    elif len(validation_losses) > len(previous_validation_losses):
        return -1.0
    else:
        return 1.0


def read_pose_corresponding_image_indexes(file_path):
    pose_corresponding_video_frame_index_array = []
    with open(file_path, "r") as filestream:
        for pose_index, line in enumerate(filestream):
            array = line.split(", ")
            array = np.array(array, dtype=np.float32)
            pose_corresponding_video_frame_index_array.append(int(array[0]))
    pose_corresponding_video_frame_index_array = np.array(pose_corresponding_video_frame_index_array, dtype=np.float32)
    return pose_corresponding_video_frame_index_array


def read_pose_corresponding_image_indexes_and_time_difference(file_path):
    pose_corresponding_video_frame_index_array = []
    pose_corresponding_video_frame_time_difference_array = []
    with open(file_path, "r") as filestream:
        for pose_index, line in enumerate(filestream):
            array = line.split(", ")
            array = np.array(array, dtype=np.float32)
            pose_corresponding_video_frame_index_array.append(int(array[0]))
            pose_corresponding_video_frame_time_difference_array.append(int(array[1]))
    pose_corresponding_video_frame_index_array = np.array(pose_corresponding_video_frame_index_array, dtype=np.int32)
    pose_corresponding_video_frame_time_difference_array = np.array(pose_corresponding_video_frame_time_difference_array, dtype=np.int32)
    return pose_corresponding_video_frame_index_array, pose_corresponding_video_frame_time_difference_array


def synchronize_selected_calibration_poses(root):
    pose_messages_path = root / "poses"
    translation_array_EM, rotation_array_EM = read_pose_messages_from_tracker(str(pose_messages_path))

    pose_image_indexes_path = root / "pose_corresponding_image_indexes"
    pose_corresponding_video_frame_index_array = read_pose_corresponding_image_indexes(str(pose_image_indexes_path))

    selected_calibration_image_name_list = list(root.glob('*.jpg'))

    # Find the most likely camera position
    for calibration_image_name in selected_calibration_image_name_list:
        calibration_image_name = str(calibration_image_name)
        difference_array = pose_corresponding_video_frame_index_array.astype(np.int32) - int(calibration_image_name[-12:-4])
        # Find if there are some zeros in it
        zero_indexes, = np.where(difference_array == 0)

        translation = np.zeros((3,), dtype=np.float64)
        rotation = np.zeros((3, 3), dtype=np.float64)
        # Average over these corresponding EM positions
        if zero_indexes.size != 0:
            flag = ""
            sum_count = 0
            for count, zero_index in enumerate(zero_indexes):
                translation += translation_array_EM[zero_index]
                rotation += rotation_array_EM[zero_index]
                sum_count = count + 1.0
            translation /= sum_count
            # print("previous", rotation / sum_count)
            if sum_count > 1.0:
                rotation = rotation_array_EM[zero_indexes[0]]
                # rotation = average_rotation(rotation / sum_count)
                # print("averaged", rotation)
        # Find the closest EM positions and use that instead
        else:
            min_indexes = np.argmin(np.abs(difference_array))
            flag = ""
            # If the closest frame are too far away, raise an error for bug inspection
            if np.amin(np.abs(difference_array)) > 10:
                flag = "bad"
                print("no best matches available for image {}".format(calibration_image_name))
                # raise OSError

            if hasattr(min_indexes, "__len__"):
                # Average over all the corresponding EM positions
                sum_count = 0
                for count, min_index in enumerate(min_indexes):
                    translation += translation_array_EM[min_index]
                    rotation += rotation_array_EM[min_index]
                    sum_count = count + 1.0
                translation /= sum_count
                # print("previous", rotation / sum_count)
                if sum_count > 1.0:
                    rotation = rotation_array_EM[min_indexes[0]]
                    # rotation = average_rotation(rotation / sum_count)
                    # print("averaged", rotation)
            else:
                translation = translation_array_EM[min_indexes]
                rotation = rotation_array_EM[min_indexes]

        with open(calibration_image_name[:-4] + flag + ".coords", "w") as filestream:
            for i in range(3):
                filestream.write("{:.5f},".format(translation[i]))
            for i in range(3):
                for j in range(3):
                    if i != 2 or j != 2:
                        filestream.write("{:.5f},".format(rotation[i][j]))
                    else:
                        filestream.write("{:.5f}\n".format(rotation[i][j]))
    return


def synchronize_image_and_poses(root, tolerance_threshold=1.0e6):
    pose_messages_path = root / "bags" / "poses_calibration"
    translation_array_EM, rotation_array_EM = read_pose_messages_from_tracker(str(pose_messages_path))

    pose_image_indexes_path = root / "bags" / "pose_corresponding_image_indexes_calibration"
    pose_corresponding_video_frame_index_array,  pose_corresponding_video_frame_time_difference_array = \
        read_pose_corresponding_image_indexes_and_time_difference(str(pose_image_indexes_path))

    best_matches_pose_indexes = np.where(pose_corresponding_video_frame_time_difference_array < tolerance_threshold)
    best_matches_pose_indexes = best_matches_pose_indexes[0]
    selected_video_frame_index_array = pose_corresponding_video_frame_index_array[best_matches_pose_indexes]

    selected_calibration_root = root / "selected_calibration_images"
    calibration_root = root / "calibration_images"
    try:
        selected_calibration_root.mkdir(mode=0o777, parents=True)
    except OSError:
        pass

    for ori_index, selected_video_frame_index in enumerate(selected_video_frame_index_array):

        dest = selected_calibration_root / "{:08d}.jpg".format(selected_video_frame_index)
        if not dest.exists():
            shutil.copyfile(str(calibration_root / "{:08d}.jpg".format(selected_video_frame_index)),
                     str(dest))

        translation = translation_array_EM[best_matches_pose_indexes[ori_index]]
        rotation = rotation_array_EM[best_matches_pose_indexes[ori_index]]
        with open(str(selected_calibration_root / "{:08d}.coords".format(selected_video_frame_index)), "w") as filestream:
            for i in range(3):
                filestream.write("{:.5f},".format(translation[i]))
            for i in range(3):
                for j in range(3):
                    if i != 2 or j != 2:
                        filestream.write("{:.5f},".format(rotation[i][j]))
                    else:
                        filestream.write("{:.5f}\n".format(rotation[i][j]))

    return


def read_camera_to_tcp_transform(root):
    transform = np.zeros((3, 4), dtype=np.float)
    with open(str(root / "camera_to_tcp"), "r") as filestream:
        for line in filestream:
            temp = line.split(" ")
            temp = np.array(temp, dtype=np.float)

    for i in range(3):
        for j in range(4):
            transform[i, j] = temp[4 * i + j]
    return transform[:, :3], transform[:, 3].reshape((3, 1))


