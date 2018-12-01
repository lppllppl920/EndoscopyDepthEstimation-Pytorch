import numpy as np
import cv2
from pathlib import Path
from plyfile import PlyData
import yaml
import transformations
import random
import torch
import json
import datetime


def get_color_file_names(root, split_ratio=[0.9, 0.05, 0.05]):
    image_list = list(root.glob('*/*/0*.jpg'))
    image_list.sort()
    split_point = [int(len(image_list) * split_ratio[0]), int(len(image_list) * (split_ratio[0] + split_ratio[1]))]
    return image_list[:split_point[0]], image_list[split_point[0]:split_point[1]], image_list[split_point[1]:]


def get_parent_folder_names(root):
    folder_list = list(root.glob('*/*/'))
    folder_list.sort()
    return folder_list


def downsample_and_crop_mask(mask, downsampling_factor, divide, suggested_h=None, suggested_w=None):
    downsampled_mask = cv2.resize(mask, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)

    # divide is related to the pooling times of the network architecture
    indexes = np.where(downsampled_mask == 255)
    h = indexes[0].max() - indexes[0].min()
    w = indexes[1].max() - indexes[1].min()

    remainder_h = h % divide
    remainder_w = w % divide

    increment_h = divide - remainder_h
    increment_w = divide - remainder_w

    start_h = max(indexes[0].min() - increment_h // 2, 0)
    if start_h == 0:
        end_h = indexes[0].max() + (increment_h - indexes[0].min())
    else:
        end_h = indexes[0].max() + (increment_h - increment_h // 2)

    start_w = max(indexes[1].min() - increment_w // 2, 0)
    if start_w == 0:
        end_w = indexes[1].max() + (increment_w - indexes[1].min())
    else:
        end_w = indexes[1].max() + (increment_w - increment_w // 2)

    if suggested_h is not None:
        if suggested_h != h:
            remain_h = suggested_h - h
            start_h = max(start_h - remain_h // 2, 0)
            end_h = suggested_h + start_h
    if suggested_w is not None:
        if suggested_w != w:
            remain_w = suggested_w - w
            start_w = max(start_w - remain_w // 2, 0)
            end_w = suggested_w + start_w

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
    with open(prefix_seq + 'visible_view_indexes') as fp:
        for line in fp:
            visible_view_indexes.append(int(line))
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
    plydata = PlyData.read(prefix_seq + "structure.ply")
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
    with open(prefix_seq + 'view_indexes_per_point') as fp:
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


def get_extrinsic_matrix_and_projection_matrix(poses, intrinsic_matrix, visible_view_count):
    projection_matrices = []
    extrinsic_matrices = []
    for i in range(visible_view_count):
        rigid_transform = transformations.quaternion_matrix(
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
    if 0.0 < inlier_percentage < 1.0:
        point_cloud_contamination_accumulator = np.zeros(len(point_cloud))
        height, width, channel = imgs[0].shape

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

            sanity_array = []
            for j in range(len(point_cloud)):
                point_3d_position = np.asarray(point_cloud[j])
                point_3d_position_camera = np.asarray(extrinsic_matrix).dot(point_3d_position)
                point_3d_position_camera = point_3d_position_camera / point_3d_position_camera[3]
                point_3d_position_camera = np.reshape(point_3d_position_camera[:3], (3,))

                point_projected_undistorted = np.asarray(projection_matrix).dot(point_3d_position)
                point_projected_undistorted = point_projected_undistorted / point_projected_undistorted[2]
                round_u = int(round(point_projected_undistorted[0]))
                round_v = int(round(point_projected_undistorted[1]))

                # We will treat this point as valid if it is projected onto the mask region
                if 0 <= round_u < width and 0 <= round_v < height:
                    if mask_boundary[round_v, round_u] > 220 and point_3d_position_camera[2] > 0.0:
                        point_to_camera_distance_2 = np.dot(point_3d_position_camera[:3], point_3d_position_camera[:3])
                        sanity_array.append(point_to_camera_distance_2 * img_hsv[round_v, round_u, 2])

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
                round_u = int(round(point_projected_undistorted[0]))
                round_v = int(round(point_projected_undistorted[1]))

                if 0 <= round_u < width and 0 <= round_v < height:
                    if mask_boundary[round_v, round_u] > 220 and point_3d_position_camera[2] > 0.0:
                        point_to_camera_distance_2 = np.dot(point_3d_position_camera[:3], point_3d_position_camera[:3])
                        sanity_value = point_to_camera_distance_2 * img_hsv[round_v, round_u, 2]
                        if sanity_value <= sanity_threshold_min or sanity_value >= sanity_threshold_max:
                            point_cloud_contamination_accumulator[j] = point_cloud_contamination_accumulator[j] + 1

        contaminated_point_cloud_indexes = []
        # TODO: How to decide on this threshold more smartly
        for i in range(point_cloud_contamination_accumulator.shape[0]):
            if point_cloud_contamination_accumulator[i] >= 10:
                contaminated_point_cloud_indexes.append(i)
    else:
        contaminated_point_cloud_indexes = []

    return contaminated_point_cloud_indexes


def get_visible_count_per_point(view_indexes_per_point):
    appearing_count = np.reshape(np.sum(view_indexes_per_point, axis=-1), (-1, 1))
    return appearing_count


def generating_pos_and_increment(idx, visible_view_indexes, adjacent_range):
    # We use the remainder of the overall idx to retrieve the visible view
    visible_view_idx = idx % len(visible_view_indexes)

    if visible_view_idx <= adjacent_range[0] - 1:
        increment = random.randint(adjacent_range[0],
                                   min(adjacent_range[1], len(visible_view_indexes) - 1 - visible_view_idx))
    elif visible_view_idx >= len(visible_view_indexes) - adjacent_range[0]:
        increment = -random.randint(adjacent_range[0], min(adjacent_range[1], visible_view_idx))

    else:
        # which direction should we increment
        direction = random.randint(0, 1)
        if direction == 1:
            increment = random.randint(adjacent_range[0],
                                       min(adjacent_range[1], len(visible_view_indexes) - 1 - visible_view_idx))
        else:
            increment = -random.randint(adjacent_range[0], min(adjacent_range[1], visible_view_idx))

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

    pair_opt_flow_images = []
    opt_flow_image_1 = np.zeros((height, width, 2), dtype=np.float32)
    opt_flow_image_2 = np.zeros((height, width, 2), dtype=np.float32)

    pair_opt_flow_mask_images = []
    opt_flow_mask_image_1 = np.zeros((height, width, 1), dtype=np.float32)
    opt_flow_mask_image_2 = np.zeros((height, width, 1), dtype=np.float32)

    point_projection_positions_1 = np.zeros((len(point_cloud) - len(contamination_point_list), 2), dtype=np.float32)
    point_projection_positions_2 = np.zeros((len(point_cloud) - len(contamination_point_list), 2), dtype=np.float32)

    # Calculate optical flows for each feature point
    # Here we compute the mean flow vector length in order to better weight these flow vectors (TODO: Do we really want this?)
    # We rule out the unreasonable flow vectors to reduce the chance of data contamination
    mean_flow_length = 0
    flow_length_count = 0
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
            round_u = int(round(point_projected_undistorted[0]))
            round_v = int(round(point_projected_undistorted[1]))

            if i == 0:
                point_projection_positions_1[count][0] = round_u
                point_projection_positions_1[count][1] = round_v

            elif i == 1:
                point_projection_positions_2[count][0] = round_u
                point_projection_positions_2[count][1] = round_v

                distance = np.abs((round_u - point_projection_positions_1[count][0]) / width) + \
                           np.abs((round_v - point_projection_positions_1[count][1]) / height)

                if distance <= 1.0:
                    mean_flow_length += distance
                    flow_length_count += 1

            count += 1

    mean_flow_length /= flow_length_count

    # Change binary optical flow mask images to weighted version to emphasize the importance of small optical flow masks
    # We rule out the unreasonable flow vectors to reduce the chance of data contamination
    # TODO: Changed it to binary mask first to see the effect on training
    # TODO: There are part of the partial occlusion for some of the clips, should we introduce visible views per point here?
    # for i in range(len(point_cloud) - len(contamination_point_list)):

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
                    opt_flow_image_1[int(v)][int(u)][0] = float(u2 - u) / width
                    opt_flow_image_1[int(v)][int(u)][1] = float(v2 - v) / height
                    opt_flow_mask_image_1[int(v)][int(u)] = 1.0 - np.exp(
                        -appearing_count_per_point[i, 0] /
                        count_weight)  # np.exp(-1.0 / (flow_factor * mean_flow_length) * distance)

        if 0 <= u2 < width and 0 <= v2 < height:
            if mask_boundary[int(v2), int(u2)] > 220:
                distance = np.abs(float(u - u2) / width) + np.abs(float(v - v2) / height)
                if distance <= 1.0:
                    opt_flow_image_2[int(v2)][int(u2)][0] = float(u - u2) / width
                    opt_flow_image_2[int(v2)][int(u2)][1] = float(v - v2) / height
                    opt_flow_mask_image_2[int(v2)][int(u2)] = 1.0 - np.exp(
                        -appearing_count_per_point[i, 0] /
                        count_weight)  # np.exp(-1.0 / (flow_factor * mean_flow_length) * distance)

        count += 1
    # TODO: To be modified
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
                round_u = int(round(point_projected_undistorted[0]))
                round_v = int(round(point_projected_undistorted[1]))
                if view_indexes_per_point[j][visible_view_indexes.index(pair_indexes[i])] > 0.5:
                    if 0 <= round_u < width and 0 <= round_v < height:
                        if mask_boundary[round_v, round_u] > 220 and point_3d_position_camera[2] > 0.0:
                            mask_img[round_v][round_u] = 1.0 - np.exp(
                                -appearing_count_per_point[j, 0] /
                                count_weight)
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
                round_u = int(round(point_projected_undistorted[0]))
                round_v = int(round(point_projected_undistorted[1]))
                if 0 <= round_u < width and 0 <= round_v < height:
                    if mask_boundary[round_v, round_u] > 220 and point_3d_position_camera[2] > 0.0:

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

    pair_opt_flow_images.append(opt_flow_image_1)
    pair_opt_flow_images.append(opt_flow_image_2)
    pair_opt_flow_images = np.array(pair_opt_flow_images, dtype="float32")
    pair_opt_flow_images = np.reshape(pair_opt_flow_images, (-1, height, width, 2))

    pair_opt_flow_mask_images.append(opt_flow_mask_image_1)
    pair_opt_flow_mask_images.append(opt_flow_mask_image_2)
    pair_opt_flow_mask_images = np.array(pair_opt_flow_mask_images, dtype="float32")
    pair_opt_flow_mask_images = np.reshape(pair_opt_flow_mask_images, (-1, height, width, 1))

    pair_mask_imgs = np.array(pair_mask_imgs, dtype="float32")
    pair_sparse_depth_imgs = np.array(pair_sparse_depth_imgs, dtype="float32")
    pair_mask_imgs = np.reshape(pair_mask_imgs, (-1, height, width, 1))
    pair_sparse_depth_imgs = np.reshape(pair_sparse_depth_imgs, (-1, height, width, 1))

    return [pair_mask_imgs, pair_sparse_depth_imgs, pair_opt_flow_mask_images, pair_opt_flow_images]


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


def save_model(model, optimizer, epoch, step, model_path):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
    }, str(model_path))
    return


def visualize_color_image(title, images, rebias=False, is_hsv=False):
    for i in range(images.shape[0]):
        image = images.data.cpu().numpy()[i]
        image = np.moveaxis(image, source=[0, 1, 2], destination=[2, 0, 1])
        if rebias:
            image = image * 0.5 + 0.5
        if is_hsv:
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR_FULL)
        cv2.imshow(title + "_" + str(i), image)


def display_depth_map(depth_map, min_value=None, max_value=None):
    if min_value is None or max_value is None:
        min_value = np.min(depth_map)
        max_value = np.max(depth_map)
    depth_map_visualize = np.abs((depth_map - min_value) / (max_value - min_value + 1.0) * 255)
    depth_map_visualize[depth_map_visualize > 255] = 255
    depth_map_visualize[depth_map_visualize <= 0.0] = 0
    depth_map_visualize = cv2.applyColorMap(np.uint8(depth_map_visualize), cv2.COLORMAP_JET)
    return depth_map_visualize


def draw_hsv(flows, title):
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
    return


def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.time().isoformat()
    log.write(unicode(json.dumps(data, sort_keys=True)))
    log.write(unicode('\n'))
    log.flush()


# Unit test
if __name__ == "__main__":
    prefix_seq = "/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData/bag_1/_start_002603_end_002984_stride_25_segment_00/"
    image_list_ = get_color_file_names(Path("/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData"))
    print(image_list_[:10])

    folder_list = get_parent_folder_names(
        Path("/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData"))
    print(folder_list)

    # mask = cv2.imread(
    #     "/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData/bag_1/_start_002603_end_002984_stride_25_segment_00/undistorted_mask.bmp",
    #     cv2.IMREAD_GRAYSCALE)
    # cropped_mask, start_h, end_h, start_w, end_w = downsample_and_crop_mask(mask, downsampling_factor=2.0,
    #                                                                         divide=2 ** (7 - 1))
    # cv2.imshow("mask", cropped_mask)
    # cv2.waitKey(100)
    #
    # stride, selected_indexes = read_selected_indexes(
    #     "/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData/bag_1/_start_002603_end_002984_stride_25_segment_00/")
    # print(stride, selected_indexes)
    #
    # visible_view_indexes = read_visible_view_indexes(
    #     "/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData/bag_1/_start_002603_end_002984_stride_25_segment_00/")
    # print(visible_view_indexes)
    #
    # intrinsics = read_camera_intrinsic_per_view(
    #     "/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData/bag_1/_start_002603_end_002984_stride_25_segment_00/")
    # print(intrinsics[0])
    #
    # poses = read_pose_data(prefix_seq)
    # print(poses)
