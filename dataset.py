'''
Author: Xingtong Liu, Ayushi Sinha, Masaru Ishii, Gregory D. Hager, Austin Reiter, Russell H. Taylor, and Mathias Unberath

Copyright (C) 2019 Johns Hopkins University - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU GENERAL PUBLIC LICENSE Version 3 license for non-commercial usage.

You should have received a copy of the GNU GENERAL PUBLIC LICENSE Version 3 license with
this file. If not, please write to: xliu89@jh.edu or rht@jhu.edu or unberath@jhu.edu
'''

import torch
import numpy as np
import cv2
import pickle
from multiprocessing import Process, Queue

from torch.utils.data import Dataset
from albumentations.pytorch.functional import img_to_tensor
import albumentations as albu

import utils


def find_largest_size(folder_list, downsampling, network_downsampling, queue_size):
    for folder in folder_list:
        # Read mask image
        undistorted_mask_boundary = cv2.imread(str(folder / "undistorted_mask.bmp"), cv2.IMREAD_GRAYSCALE)
        # Downsample and crop the undistorted mask image
        _, start_h, end_h, start_w, end_w = \
            utils.downsample_and_crop_mask(undistorted_mask_boundary, downsampling_factor=downsampling,
                                           divide=network_downsampling)
        queue_size.put([end_h - start_h, end_w - start_w])


def pre_processing_data(process_id, folder_list, downsampling, network_downsampling, is_hsv, inlier_percentage,
                        visible_interval,
                        suggested_h, suggested_w,
                        queue_clean_point_list, queue_intrinsic_matrix, queue_point_cloud,
                        queue_mask_boundary, queue_view_indexes_per_point, queue_selected_indexes,
                        queue_visible_view_indexes,
                        queue_extrinsics, queue_projection, queue_crop_positions, queue_estimated_scale):
    for folder in folder_list:
        # We use folder path as the key for dictionaries
        # Read undistorted mask image
        folder_str = str(folder)
        undistorted_mask_boundary = cv2.imread(str(folder / "undistorted_mask.bmp"), cv2.IMREAD_GRAYSCALE)
        # Downsample and crop the undistorted mask image
        cropped_downsampled_undistorted_mask_boundary, start_h, end_h, start_w, end_w = \
            utils.downsample_and_crop_mask(undistorted_mask_boundary, downsampling_factor=downsampling,
                                           divide=network_downsampling, suggested_h=suggested_h,
                                           suggested_w=suggested_w)
        queue_mask_boundary.put([folder_str, cropped_downsampled_undistorted_mask_boundary])
        queue_crop_positions.put([folder_str, [start_h, end_h, start_w, end_w]])
        # Read selected image indexes and stride
        stride, selected_indexes = utils.read_selected_indexes(folder)
        queue_selected_indexes.put([folder_str, selected_indexes])
        # Read visible view indexes
        visible_view_indexes = utils.read_visible_view_indexes(folder)
        queue_visible_view_indexes.put([folder_str, visible_view_indexes])
        # Read undistorted camera intrinsics
        undistorted_camera_intrinsic_per_view = utils.read_camera_intrinsic_per_view(folder)
        # Downsample and crop the undistorted camera intrinsics
        # Assuming for now that camera intrinsics within each clip remains the same
        cropped_downsampled_undistorted_intrinsic_matrix = utils.modify_camera_intrinsic_matrix(
            undistorted_camera_intrinsic_per_view[0], start_h=start_h,
            start_w=start_w, downsampling_factor=downsampling)
        queue_intrinsic_matrix.put([folder_str, cropped_downsampled_undistorted_intrinsic_matrix])
        # Read sparse point cloud from SfM
        point_cloud = utils.read_point_cloud(str(folder / "structure.ply"))
        queue_point_cloud.put([folder_str, point_cloud])
        # self.point_cloud_per_seq[folder] = point_cloud
        # Read visible view indexes per point
        view_indexes_per_point = utils.read_view_indexes_per_point(folder, visible_view_indexes=
        visible_view_indexes, point_cloud_count=len(point_cloud))
        # Update view_indexes_per_point_per_seq with neighborhood frames to increase stability and
        # avoid as much occlusion problem as possible
        view_indexes_per_point = utils.overlapping_visible_view_indexes_per_point(view_indexes_per_point,
                                                                                  visible_interval)
        queue_view_indexes_per_point.put([folder_str, view_indexes_per_point])
        # Read pose data for all visible views
        poses = utils.read_pose_data(folder)
        # Calculate extrinsic and projection matrices
        visible_extrinsic_matrices, visible_cropped_downsampled_undistorted_projection_matrices = \
            utils.get_extrinsic_matrix_and_projection_matrix(poses,
                                                             intrinsic_matrix=
                                                             cropped_downsampled_undistorted_intrinsic_matrix,
                                                             visible_view_count=len(visible_view_indexes))
        queue_extrinsics.put([folder_str, visible_extrinsic_matrices])
        queue_projection.put([folder_str, visible_cropped_downsampled_undistorted_projection_matrices])
        # Get approximate data global scale to reduce training data imbalance
        global_scale = utils.global_scale_estimation(visible_extrinsic_matrices, point_cloud)
        queue_estimated_scale.put([folder_str, global_scale])
        visible_cropped_downsampled_imgs = utils.get_color_imgs(folder, visible_view_indexes=visible_view_indexes,
                                                                start_h=start_h, start_w=start_w,
                                                                end_h=end_h, end_w=end_w,
                                                                downsampling_factor=downsampling,
                                                                is_hsv=is_hsv)
        # Calculate contaminated point list
        clean_point_indicator_array = utils.get_clean_point_list(imgs=visible_cropped_downsampled_imgs,
                                                                 point_cloud=point_cloud,
                                                                 mask_boundary=
                                                                 cropped_downsampled_undistorted_mask_boundary,
                                                                 inlier_percentage=inlier_percentage,
                                                                 projection_matrices=
                                                                 visible_cropped_downsampled_undistorted_projection_matrices,
                                                                 extrinsic_matrices=visible_extrinsic_matrices,
                                                                 is_hsv=is_hsv,
                                                                 view_indexes_per_point=view_indexes_per_point)
        queue_clean_point_list.put([folder_str, clean_point_indicator_array])
        print("sequence {} finished".format(folder_str))

    print("{}th process finished".format(process_id))


class SfMDataset(Dataset):
    def __init__(self, image_file_names, folder_list, adjacent_range,
                 transform, downsampling, network_downsampling, inlier_percentage, visible_interval,
                 use_store_data, store_data_root, phase, is_hsv, num_pre_workers, rgb_mode, num_iter=None):
        self.rgb_mode = rgb_mode
        self.image_file_names = image_file_names
        self.folder_list = folder_list
        self.transform = transform
        assert (len(adjacent_range) == 2)
        self.adjacent_range = adjacent_range
        self.transform = transform
        self.is_hsv = is_hsv
        self.inlier_percentage = inlier_percentage
        self.downsampling = downsampling
        self.network_downsampling = network_downsampling
        self.phase = phase
        self.visible_interval = visible_interval
        self.num_pre_workers = min(len(folder_list), num_pre_workers)
        self.num_iter = num_iter
        self.num_sample = len(self.image_file_names)

        self.clean_point_list_per_seq = {}
        self.intrinsic_matrix_per_seq = {}
        self.point_cloud_per_seq = {}
        self.mask_boundary_per_seq = {}
        self.view_indexes_per_point_per_seq = {}
        self.selected_indexes_per_seq = {}
        self.visible_view_indexes_per_seq = {}
        self.extrinsics_per_seq = {}
        self.projection_per_seq = {}
        self.crop_positions_per_seq = {}
        self.estimated_scale_per_seq = {}
        self.normalize = albu.Normalize(std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5), max_pixel_value=255.0)

        if phase == "Evaluation":
            precompute_path = store_data_root / ("evaluate_precompute_" + str(
                self.downsampling) + "_" + str(self.network_downsampling) + "_" + str(self.inlier_percentage) + ".pkl")
        else:
            precompute_path = store_data_root / ("precompute_" + str(
                self.downsampling) + "_" + str(self.network_downsampling) + "_" + str(self.inlier_percentage) + ".pkl")

        # Save all intermediate results to hard disk for quick access later on
        if not use_store_data or not precompute_path.exists():
            queue_size = Queue()
            queue_clean_point_list = Queue()
            queue_intrinsic_matrix = Queue()
            queue_point_cloud = Queue()
            queue_mask_boundary = Queue()
            queue_view_indexes_per_point = Queue()
            queue_selected_indexes = Queue()
            queue_visible_view_indexes = Queue()
            queue_extrinsics = Queue()
            queue_projection = Queue()
            queue_crop_positions = Queue()
            queue_estimated_scale = Queue()

            process_pool = []

            interval = len(self.folder_list) / self.num_pre_workers

            # Go through the entire image list to find the largest required h and w
            for i in range(self.num_pre_workers):
                process_pool.append(Process(target=find_largest_size, args=(
                    self.folder_list[
                    int(np.round(i * interval)): min(int(np.round((i + 1) * interval)), len(self.folder_list))],
                    self.downsampling,
                    self.network_downsampling,
                    queue_size)))

            for t in process_pool:
                t.start()

            largest_h = 0
            largest_w = 0

            for t in process_pool:
                while t.is_alive():
                    while not queue_size.empty():
                        h, w = queue_size.get()
                        if h > largest_h:
                            largest_h = h
                        if w > largest_w:
                            largest_w = w
                    t.join(timeout=1)

            while not queue_size.empty():
                h, w = queue_size.get()
                if h > largest_h:
                    largest_h = h
                if w > largest_w:
                    largest_w = w

            if largest_h == 0 or largest_w == 0:
                print("image size calculation failed.")
                raise IOError

            print("Largest image size is: ", largest_h, largest_w)
            print("Start pre-processing dataset...")
            process_pool = []
            for i in range(self.num_pre_workers):
                process_pool.append(Process(target=pre_processing_data,
                                            args=(i, self.folder_list[
                                                     int(np.round(i * interval)): min(int(np.round((i + 1) * interval)),
                                                                                      len(self.folder_list))],
                                                  self.downsampling, self.network_downsampling, self.is_hsv,
                                                  self.inlier_percentage, self.visible_interval, largest_h, largest_w,
                                                  queue_clean_point_list,
                                                  queue_intrinsic_matrix, queue_point_cloud,
                                                  queue_mask_boundary, queue_view_indexes_per_point,
                                                  queue_selected_indexes,
                                                  queue_visible_view_indexes,
                                                  queue_extrinsics, queue_projection,
                                                  queue_crop_positions,
                                                  queue_estimated_scale)))

            for t in process_pool:
                t.start()

            count = 0
            for t in process_pool:
                print("Waiting for {:d}th process to complete".format(count))
                count += 1
                while t.is_alive():
                    while not queue_selected_indexes.empty():
                        folder, selected_indexes = queue_selected_indexes.get()
                        self.selected_indexes_per_seq[folder] = selected_indexes
                    while not queue_visible_view_indexes.empty():
                        folder, visible_view_indexes = queue_visible_view_indexes.get()
                        self.visible_view_indexes_per_seq[folder] = visible_view_indexes
                    while not queue_view_indexes_per_point.empty():
                        folder, view_indexes_per_point = queue_view_indexes_per_point.get()
                        self.view_indexes_per_point_per_seq[folder] = view_indexes_per_point
                    while not queue_clean_point_list.empty():
                        folder, clean_point_list = queue_clean_point_list.get()
                        self.clean_point_list_per_seq[folder] = clean_point_list
                    while not queue_intrinsic_matrix.empty():
                        folder, intrinsic_matrix = queue_intrinsic_matrix.get()
                        self.intrinsic_matrix_per_seq[folder] = intrinsic_matrix
                    while not queue_extrinsics.empty():
                        folder, extrinsics = queue_extrinsics.get()
                        self.extrinsics_per_seq[folder] = extrinsics
                    while not queue_projection.empty():
                        folder, projection = queue_projection.get()
                        self.projection_per_seq[folder] = projection
                    while not queue_crop_positions.empty():
                        folder, crop_positions = queue_crop_positions.get()
                        self.crop_positions_per_seq[folder] = crop_positions
                    while not queue_point_cloud.empty():
                        folder, point_cloud = queue_point_cloud.get()
                        self.point_cloud_per_seq[folder] = point_cloud
                    while not queue_mask_boundary.empty():
                        folder, mask_boundary = queue_mask_boundary.get()
                        self.mask_boundary_per_seq[folder] = mask_boundary
                    while not queue_estimated_scale.empty():
                        folder, estiamted_scale = queue_estimated_scale.get()
                        self.estimated_scale_per_seq[folder] = estiamted_scale
                    t.join(timeout=1)

            while not queue_selected_indexes.empty():
                folder, selected_indexes = queue_selected_indexes.get()
                self.selected_indexes_per_seq[folder] = selected_indexes
            while not queue_visible_view_indexes.empty():
                folder, visible_view_indexes = queue_visible_view_indexes.get()
                self.visible_view_indexes_per_seq[folder] = visible_view_indexes
            while not queue_view_indexes_per_point.empty():
                folder, view_indexes_per_point = queue_view_indexes_per_point.get()
                self.view_indexes_per_point_per_seq[folder] = view_indexes_per_point
            while not queue_clean_point_list.empty():
                folder, clean_point_list = queue_clean_point_list.get()
                self.clean_point_list_per_seq[folder] = clean_point_list
            while not queue_intrinsic_matrix.empty():
                folder, intrinsic_matrix = queue_intrinsic_matrix.get()
                self.intrinsic_matrix_per_seq[folder] = intrinsic_matrix
            while not queue_extrinsics.empty():
                folder, extrinsics = queue_extrinsics.get()
                self.extrinsics_per_seq[folder] = extrinsics
            while not queue_projection.empty():
                folder, projection = queue_projection.get()
                self.projection_per_seq[folder] = projection
            while not queue_crop_positions.empty():
                folder, crop_positions = queue_crop_positions.get()
                self.crop_positions_per_seq[folder] = crop_positions
            while not queue_point_cloud.empty():
                folder, point_cloud = queue_point_cloud.get()
                self.point_cloud_per_seq[folder] = point_cloud
            while not queue_mask_boundary.empty():
                folder, mask_boundary = queue_mask_boundary.get()
                self.mask_boundary_per_seq[folder] = mask_boundary
            while not queue_estimated_scale.empty():
                folder, estimated_scale = queue_estimated_scale.get()
                self.estimated_scale_per_seq[folder] = estimated_scale
            print("Pre-processing complete.")

            # Store all intermediate information to a single data file
            with open(str(precompute_path), "wb") as f:
                pickle.dump(
                    [self.crop_positions_per_seq, self.selected_indexes_per_seq,
                     self.visible_view_indexes_per_seq,
                     self.point_cloud_per_seq, self.intrinsic_matrix_per_seq,
                     self.mask_boundary_per_seq, self.view_indexes_per_point_per_seq, self.extrinsics_per_seq,
                     self.projection_per_seq, self.clean_point_list_per_seq,
                     self.downsampling, self.network_downsampling, self.inlier_percentage,
                     self.estimated_scale_per_seq],
                    f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(str(precompute_path), "rb") as f:
                [self.crop_positions_per_seq, self.selected_indexes_per_seq,
                 self.visible_view_indexes_per_seq,
                 self.point_cloud_per_seq, self.intrinsic_matrix_per_seq,
                 self.mask_boundary_per_seq, self.view_indexes_per_point_per_seq, self.extrinsics_per_seq,
                 self.projection_per_seq, self.clean_point_list_per_seq,
                 self.downsampling, self.network_downsampling,
                 self.inlier_percentage, self.estimated_scale_per_seq] = pickle.load(f)

    def __len__(self):
        if self.num_iter is None:
            return len(self.image_file_names)
        else:
            return self.num_iter

    def __getitem__(self, idx):
        if self.phase == 'train' or self.phase == 'validation':
            while True:
                img_file_name = self.image_file_names[idx % self.num_sample]
                # Retrieve the folder path
                folder = str(img_file_name.parent)
                # Randomly pick one adjacent frame
                # We assume the filename has 8 logits followed by ".jpg"
                start_h, end_h, start_w, end_w = self.crop_positions_per_seq[folder]
                pos, increment = utils.generating_pos_and_increment(idx=idx,
                                                                    visible_view_indexes=
                                                                    self.visible_view_indexes_per_seq[
                                                                        folder],
                                                                    adjacent_range=self.adjacent_range)
                img_file_name = self.visible_view_indexes_per_seq[folder][
                    idx % len(self.visible_view_indexes_per_seq[folder])]

                # Get pair visible view indexes and pair extrinsic and projection matrices
                pair_indexes = [self.visible_view_indexes_per_seq[folder][pos],
                                self.visible_view_indexes_per_seq[folder][pos + increment]]
                pair_extrinsic_matrices = [self.extrinsics_per_seq[folder][pos],
                                           self.extrinsics_per_seq[folder][pos + increment]]
                pair_projection_matrices = [self.projection_per_seq[folder][pos],
                                            self.projection_per_seq[folder][pos + increment]]

                pair_mask_imgs, pair_sparse_depth_imgs, pair_flow_mask_imgs, pair_flow_imgs = \
                    utils.get_torch_training_data(pair_extrinsics=pair_extrinsic_matrices,
                                                  pair_projections=
                                                  pair_projection_matrices, pair_indexes=pair_indexes,
                                                  point_cloud=self.point_cloud_per_seq[folder],
                                                  mask_boundary=self.mask_boundary_per_seq[folder],
                                                  view_indexes_per_point=self.view_indexes_per_point_per_seq[folder],
                                                  visible_view_indexes=self.visible_view_indexes_per_seq[folder],
                                                  clean_point_list=self.clean_point_list_per_seq[
                                                      folder])

                if np.sum(pair_mask_imgs[0]) != 0 and np.sum(pair_mask_imgs[1]) != 0:
                    break
                else:
                    idx = np.random.randint(0, len(self.image_file_names))

            # Read pair images with downsampling and cropping
            pair_imgs = utils.get_pair_color_imgs(prefix_seq=folder, pair_indexes=pair_indexes, start_h=start_h,
                                                  start_w=start_w,
                                                  end_h=end_h, end_w=end_w, downsampling_factor=self.downsampling,
                                                  is_hsv=self.is_hsv, rgb_mode=self.rgb_mode)

            # Calculate relative motion between two frames
            relative_motion = np.matmul(pair_extrinsic_matrices[0], np.linalg.inv(pair_extrinsic_matrices[1]))
            rotation_1_wrt_2 = np.reshape(relative_motion[:3, :3], (3, 3)).astype(np.float32)
            translation_1_wrt_2 = (
                    np.reshape(relative_motion[:3, 3], (3, 1)) / self.estimated_scale_per_seq[folder]).astype(
                np.float32)

            # Scale the sparse depth map
            pair_sparse_depth_imgs[0] /= self.estimated_scale_per_seq[folder]
            pair_sparse_depth_imgs[1] /= self.estimated_scale_per_seq[folder]

            # Format training data
            color_img_1 = pair_imgs[0]
            color_img_2 = pair_imgs[1]

            rotation_2_wrt_1 = np.transpose(rotation_1_wrt_2).astype(np.float32)
            translation_2_wrt_1 = np.matmul(-np.transpose(rotation_1_wrt_2), translation_1_wrt_2).astype(np.float32)

            rotation_1_wrt_2 = rotation_1_wrt_2.reshape((3, 3))
            rotation_2_wrt_1 = rotation_2_wrt_1.reshape((3, 3))
            translation_1_wrt_2 = translation_1_wrt_2.reshape((3, 1))
            translation_2_wrt_1 = translation_2_wrt_1.reshape((3, 1))

            sparse_depth_img_1 = pair_sparse_depth_imgs[0].astype(np.float32)
            sparse_depth_img_2 = pair_sparse_depth_imgs[1].astype(np.float32)
            mask_img_1 = pair_mask_imgs[0].astype(np.float32)
            mask_img_2 = pair_mask_imgs[1].astype(np.float32)
            sparse_depth_img_1 = sparse_depth_img_1.reshape((sparse_depth_img_1.shape[0],
                                                             sparse_depth_img_1.shape[1], 1))
            sparse_depth_img_2 = sparse_depth_img_2.reshape((sparse_depth_img_2.shape[0],
                                                             sparse_depth_img_2.shape[1], 1))
            mask_img_1 = mask_img_1.reshape(
                (mask_img_1.shape[0], mask_img_1.shape[1], 1))
            mask_img_2 = mask_img_2.reshape(
                (mask_img_2.shape[0], mask_img_2.shape[1], 1))
            flow_mask_img_1 = pair_flow_mask_imgs[0].astype(np.float32)
            flow_mask_img_2 = pair_flow_mask_imgs[1].astype(np.float32)
            flow_img_1 = pair_flow_imgs[0].astype(np.float32)
            flow_img_2 = pair_flow_imgs[1].astype(np.float32)

            intrinsic_matrix = self.intrinsic_matrix_per_seq[folder][:3, :3]
            intrinsic_matrix = intrinsic_matrix.astype(np.float32)
            intrinsic_matrix = intrinsic_matrix.reshape((3, 3))

            mask_boundary = self.mask_boundary_per_seq[folder].astype(np.float32) / 255.0
            mask_boundary[mask_boundary > 0.9] = 1.0
            mask_boundary[mask_boundary <= 0.9] = 0.0
            mask_boundary = mask_boundary.reshape((mask_boundary.shape[0], mask_boundary.shape[1], 1))

            if self.phase == 'train':
                if self.transform is not None:
                    if self.is_hsv:
                        color_img_1 = cv2.cvtColor(np.uint8(color_img_1), cv2.COLOR_HSV2RGB_FULL)
                        color_img_2 = cv2.cvtColor(np.uint8(color_img_2), cv2.COLOR_HSV2RGB_FULL)
                    # Data augmentation
                    color_img_1 = self.transform(image=color_img_1)['image']
                    color_img_2 = self.transform(image=color_img_2)['image']
                    if self.is_hsv:
                        color_img_1 = cv2.cvtColor(np.uint8(color_img_1),
                                                   cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
                        color_img_2 = cv2.cvtColor(np.uint8(color_img_2),
                                                   cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
                # Normalize
                color_img_1 = self.normalize(image=color_img_1)['image']
                color_img_2 = self.normalize(image=color_img_2)['image']
            else:
                # Normalize
                color_img_1 = self.normalize(image=color_img_1)['image']
                color_img_2 = self.normalize(image=color_img_2)['image']

            return [img_to_tensor(color_img_1), img_to_tensor(color_img_2),
                    img_to_tensor(sparse_depth_img_1), img_to_tensor(sparse_depth_img_2),
                    img_to_tensor(mask_img_1), img_to_tensor(mask_img_2),
                    img_to_tensor(flow_img_1), img_to_tensor(flow_img_2),
                    img_to_tensor(flow_mask_img_1), img_to_tensor(flow_mask_img_2),
                    img_to_tensor(mask_boundary),
                    torch.from_numpy(rotation_1_wrt_2),
                    torch.from_numpy(rotation_2_wrt_1), torch.from_numpy(translation_1_wrt_2),
                    torch.from_numpy(translation_2_wrt_1), torch.from_numpy(intrinsic_matrix),
                    folder, img_file_name]

        elif self.phase == 'test':
            img_file_name = self.image_file_names[idx]
            # Retrieve the folder path
            folder = str(img_file_name.parent)
            start_h, end_h, start_w, end_w = self.crop_positions_per_seq[folder]
            color_img_1 = utils.get_test_color_img(str(img_file_name), start_h, end_h, start_w, end_w,
                                                   self.downsampling, self.is_hsv, rgb_mode=self.rgb_mode)
            # Normalize
            color_img_1 = self.normalize(image=color_img_1)['image']

            intrinsic_matrix = self.intrinsic_matrix_per_seq[folder][:3, :3]
            intrinsic_matrix = intrinsic_matrix.astype(np.float32)
            intrinsic_matrix = intrinsic_matrix.reshape((3, 3))

            mask_boundary = self.mask_boundary_per_seq[folder].astype(np.float32) / 255.0
            mask_boundary[mask_boundary > 0.9] = 1.0
            mask_boundary[mask_boundary <= 0.9] = 0.0
            mask_boundary = mask_boundary.reshape((mask_boundary.shape[0], mask_boundary.shape[1], 1))

            return [img_to_tensor(color_img_1),
                    img_to_tensor(mask_boundary),
                    torch.from_numpy(intrinsic_matrix),
                    img_file_name.name[-12:-4]]
