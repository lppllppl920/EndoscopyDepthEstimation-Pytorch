import torch
import numpy as np
import cv2
import pickle
import random
import Queue
from multiprocessing import Process, Queue

from torch.utils.data import Dataset
from albumentations.pytorch.functional import img_to_tensor

import utils


def find_largest_size(folder_list, downsampling, net_depth, queue_size):
    for folder in folder_list:
        folder = str(folder) + "/"
        # Read undistorted mask image
        undistorted_mask_boundary = cv2.imread(folder + "undistorted_mask.bmp", cv2.IMREAD_GRAYSCALE)
        # Downsample and crop the undistorted mask image
        _, start_h, end_h, start_w, end_w = \
            utils.downsample_and_crop_mask(undistorted_mask_boundary, downsampling_factor=downsampling,
                                           divide=2 ** (net_depth - 1))
        queue_size.put([end_h - start_h, end_w - start_w])


def pre_processing_data(folder_list, downsampling, net_depth, is_hsv, inlier_percentage, suggested_h, suggested_w,
                        queue_contamination_point_list, queue_appearing_count,
                        queue_intrinsic_matrix, queue_point_cloud,
                        queue_mask_boundary, queue_view_indexes_per_point, queue_selected_indexes,
                        queue_visible_view_indexes,
                        queue_extrinsics, queue_projection, queue_crop_positions):
    for folder in folder_list:
        # Retrieve folder path
        folder = str(folder) + "/"
        # Pre-build all data
        # We use folder path as the key for dictionaries
        print(folder)
        # Read undistorted mask image
        undistorted_mask_boundary = cv2.imread(folder + "undistorted_mask.bmp", cv2.IMREAD_GRAYSCALE)
        # Downsample and crop the undistorted mask image
        cropped_downsampled_undistorted_mask_boundary, start_h, end_h, start_w, end_w = \
            utils.downsample_and_crop_mask(undistorted_mask_boundary, downsampling_factor=downsampling,
                                           divide=2 ** (net_depth - 1), suggested_h=suggested_h,
                                           suggested_w=suggested_w)
        queue_mask_boundary.put([folder, cropped_downsampled_undistorted_mask_boundary])
        queue_crop_positions.put([folder, [start_h, end_h, start_w, end_w]])
        # Read selected image indexes and stride
        stride, selected_indexes = utils.read_selected_indexes(folder)
        queue_selected_indexes.put([folder, selected_indexes])
        # Read visible view indexes
        visible_view_indexes = utils.read_visible_view_indexes(folder)
        queue_visible_view_indexes.put([folder, visible_view_indexes])
        # Read undistorted camera intrinsics
        undistorted_camera_intrinsic_per_view = utils.read_camera_intrinsic_per_view(folder)
        # Downsample and crop the undistorted camera intrinsics
        # Assuming for now that camera intrinsics within each clip remains the same
        cropped_downsampled_undistorted_intrinsic_matrix = utils.modify_camera_intrinsic_matrix(
            undistorted_camera_intrinsic_per_view[0], start_h=start_h,
            start_w=start_w, downsampling_factor=downsampling)
        queue_intrinsic_matrix.put([folder, cropped_downsampled_undistorted_intrinsic_matrix])
        # Read sparse point cloud from SfM
        point_cloud = utils.read_point_cloud(folder)
        queue_point_cloud.put([folder, point_cloud])
        # self.point_cloud_per_seq[folder] = point_cloud
        # Read visible view indexes per point
        view_indexes_per_point = utils.read_view_indexes_per_point(folder, visible_view_indexes=
        visible_view_indexes, point_cloud_count=len(point_cloud))
        queue_view_indexes_per_point.put([folder, view_indexes_per_point])
        # Read pose data for all visible views
        poses = utils.read_pose_data(folder)
        # Calculate extrinsic and projection matrices
        visible_extrinsic_matrices, visible_cropped_downsampled_undistorted_projection_matrices = \
            utils.get_extrinsic_matrix_and_projection_matrix(poses,
                                                             intrinsic_matrix=
                                                             cropped_downsampled_undistorted_intrinsic_matrix,
                                                             visible_view_count=len(visible_view_indexes))
        queue_extrinsics.put([folder, visible_extrinsic_matrices])
        queue_projection.put([folder, visible_cropped_downsampled_undistorted_projection_matrices])

        visible_cropped_downsampled_imgs = utils.get_color_imgs(folder, visible_view_indexes=
        visible_view_indexes,
                                                                start_h=start_h, start_w=start_w,
                                                                end_h=end_h, end_w=end_w,
                                                                downsampling_factor=downsampling,
                                                                is_hsv=is_hsv)
        # Calculate contaminated point list
        contaminated_point_list = utils.get_contaminated_point_list(imgs=visible_cropped_downsampled_imgs,
                                                                    point_cloud=point_cloud,
                                                                    mask_boundary=
                                                                    cropped_downsampled_undistorted_mask_boundary,
                                                                    inlier_percentage=inlier_percentage,
                                                                    projection_matrices=
                                                                    visible_cropped_downsampled_undistorted_projection_matrices,
                                                                    extrinsic_matrices=visible_extrinsic_matrices,
                                                                    is_hsv=is_hsv)
        queue_contamination_point_list.put([folder, contaminated_point_list])
        # Calculate appearing count of view per 3D point
        appearing_count_per_point = utils.get_visible_count_per_point(
            view_indexes_per_point=view_indexes_per_point)
        queue_appearing_count.put([folder, appearing_count_per_point])


class SfMDataset(Dataset):
    def __init__(self, image_file_names, folder_list, split_ratio, adjacent_range=[1, 10], to_augment=True,
                 transform=None,
                 downsampling=1.0, net_depth=6, inlier_percentage=0.99, use_store_data=False, store_data_root=None,
                 use_view_indexes_per_point=True, visualize=False, phase="train",
                 is_hsv=True, use_random_seed=False):
        self.image_file_names = image_file_names
        self.folder_list = folder_list
        self.split_ratio = split_ratio
        assert (len(split_ratio) == 3)
        self.to_augment = to_augment
        self.transform = transform
        self.adjacent_range = adjacent_range
        assert (len(adjacent_range) == 2)
        self.transform = transform
        self.to_augment = to_augment
        self.use_view_indexes_per_point = use_view_indexes_per_point
        self.visualize = visualize
        self.is_hsv = is_hsv
        self.use_random_seed = use_random_seed
        if use_random_seed:
            random.seed(10086)
        # Save all intermediate results to hard disk for quick access later on
        self.inlier_percentage = inlier_percentage
        self.downsampling = downsampling
        self.net_depth = net_depth

        self.contamination_point_list_per_seq = {}
        self.appearing_count_per_seq = {}
        self.intrinsic_matrix_per_seq = {}
        self.point_cloud_per_seq = {}
        self.mask_boundary_per_seq = {}
        self.view_indexes_per_point_per_seq = {}
        self.selected_indexes_per_seq = {}
        self.visible_view_indexes_per_seq = {}
        self.extrinsics_per_seq = {}
        self.projection_per_seq = {}
        self.crop_positions_per_seq = {}

        if not use_store_data:
            queue_size = Queue()
            queue_contamination_point_list = Queue()
            queue_appearing_count = Queue()
            queue_intrinsic_matrix = Queue()
            queue_point_cloud = Queue()
            queue_mask_boundary = Queue()
            queue_view_indexes_per_point = Queue()
            queue_selected_indexes = Queue()
            queue_visible_view_indexes = Queue()
            queue_extrinsics = Queue()
            queue_projection = Queue()
            queue_crop_positions = Queue()

            process_pool = []

            workers = 8
            interval = len(self.folder_list) // workers

            # Go through the entire image list to find the largest required h and w
            for i in range(workers - 1):
                process_pool.append(Process(target=find_largest_size, args=(
                    self.folder_list[i * interval: (i + 1) * interval], self.downsampling, self.net_depth,
                    queue_size)))
            process_pool.append(Process(target=find_largest_size, args=(
                self.folder_list[(workers - 1) * interval:], self.downsampling, self.net_depth, queue_size)))
            for t in process_pool:
                t.start()
            for t in process_pool:
                t.join()

            largest_h = 0
            largest_w = 0
            while not queue_size.empty():
                h, w = queue_size.get()
                if h > largest_h:
                    largest_h = h
                if w > largest_w:
                    largest_w = w
            print ("Largest image size is: ", largest_h, largest_w)
            cv2.waitKey()
            process_pool = []
            for i in range(workers - 1):
                process_pool.append(Process(target=pre_processing_data,
                                            args=(self.folder_list[i * interval: (i + 1) * interval],
                                                  self.downsampling, self.net_depth, self.is_hsv,
                                                  self.inlier_percentage, largest_h, largest_w,
                                                  queue_contamination_point_list, queue_appearing_count,
                                                  queue_intrinsic_matrix, queue_point_cloud,
                                                  queue_mask_boundary, queue_view_indexes_per_point,
                                                  queue_selected_indexes,
                                                  queue_visible_view_indexes,
                                                  queue_extrinsics, queue_projection,
                                                  queue_crop_positions)))
                process_pool.append(Process(target=pre_processing_data,
                                            args=(self.folder_list[(workers - 1) * interval:],
                                                  self.downsampling, self.net_depth, self.is_hsv,
                                                  self.inlier_percentage, largest_h, largest_w,
                                                  queue_contamination_point_list, queue_appearing_count,
                                                  queue_intrinsic_matrix, queue_point_cloud,
                                                  queue_mask_boundary, queue_view_indexes_per_point,
                                                  queue_selected_indexes,
                                                  queue_visible_view_indexes,
                                                  queue_extrinsics, queue_projection,
                                                  queue_crop_positions)))

            for t in process_pool:
                t.start()
            for t in process_pool:
                t.join()

            # Gathering generated data
            while not queue_selected_indexes.empty():
                folder, selected_indexes = queue_selected_indexes.get()
                self.selected_indexes_per_seq[folder] = selected_indexes
            while not queue_visible_view_indexes.empty():
                folder, visible_view_indexes = queue_visible_view_indexes.get()
                self.visible_view_indexes_per_seq[folder] = visible_view_indexes
            while not queue_view_indexes_per_point.empty():
                folder, view_indexes_per_point = queue_view_indexes_per_point.get()
                self.view_indexes_per_point_per_seq[folder] = view_indexes_per_point
            while not queue_contamination_point_list.empty():
                folder, contamination_point_list = queue_contamination_point_list.get()
                self.contamination_point_list_per_seq[folder] = contamination_point_list
            while not queue_appearing_count.empty():
                folder, appearing_count = queue_appearing_count.get()
                self.appearing_count_per_seq[folder] = appearing_count
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

            # for img_file_name in self.image_file_names:
            #     img_file_name = str(img_file_name)
            #     # Retrieve folder path
            #     folder = img_file_name[:-12]
            #
            #     # Pre-build all data
            #     if folder not in self.selected_indexes_per_seq.keys():
            #         # We use folder path as the key for dictionaries
            #         print(folder)
            #         # Read undistorted mask image
            #         undistorted_mask_boundary = cv2.imread(folder + "undistorted_mask.bmp", cv2.IMREAD_GRAYSCALE)
            #         # Downsample and crop the undistorted mask image
            #         cropped_downsampled_undistorted_mask_boundary, start_h, end_h, start_w, end_w = \
            #             utils.downsample_and_crop_mask(undistorted_mask_boundary, downsampling_factor=self.downsampling,
            #                                            divide=2 ** (self.net_depth - 1))
            #         self.mask_boundary_per_seq[folder] = cropped_downsampled_undistorted_mask_boundary
            #         self.crop_positions_per_seq[folder] = [start_h, end_h, start_w, end_w]
            #         # Read selected image indexes and stride
            #         stride, selected_indexes = utils.read_selected_indexes(folder)
            #         self.selected_indexes_per_seq[folder] = selected_indexes
            #         # Read visible view indexes
            #         visible_view_indexes = utils.read_visible_view_indexes(folder)
            #         self.visible_view_indexes_per_seq[folder] = visible_view_indexes
            #         # Read undistorted camera intrinsics
            #         undistorted_camera_intrinsic_per_view = utils.read_camera_intrinsic_per_view(folder)
            #         # Downsample and crop the undistorted camera intrinsics
            #         # Assuming for now that camera intrinsics within each clip remains the same
            #         cropped_downsampled_undistorted_intrinsic_matrix = utils.modify_camera_intrinsic_matrix(
            #             undistorted_camera_intrinsic_per_view[0], start_h=start_h,
            #             start_w=start_w, downsampling_factor=self.downsampling)
            #         self.intrinsic_matrix_per_seq[folder] = cropped_downsampled_undistorted_intrinsic_matrix
            #         # Read sparse point cloud from SfM
            #         point_cloud = utils.read_point_cloud(folder)
            #         self.point_cloud_per_seq[folder] = point_cloud
            #         # Read visible view indexes per point
            #         view_indexes_per_point = utils.read_view_indexes_per_point(folder, visible_view_indexes=
            #         self.visible_view_indexes_per_seq[folder],
            #                                                                    point_cloud_count=len(
            #                                                                        self.point_cloud_per_seq[folder]))
            #         self.view_indexes_per_point_per_seq[folder] = view_indexes_per_point
            #         # Read pose data for all visible views
            #         poses = utils.read_pose_data(folder)
            #         # Calculate extrinsic and projection matrices
            #         visible_extrinsic_matrices, visible_cropped_downsampled_undistorted_projection_matrices = \
            #             utils.get_extrinsic_matrix_and_projection_matrix(poses,
            #                                                              intrinsic_matrix=self.intrinsic_matrix_per_seq[
            #                                                                  folder],
            #                                                              visible_view_count=len(
            #                                                                  self.visible_view_indexes_per_seq[folder]))
            #         self.extrinsics_per_seq[folder] = visible_extrinsic_matrices
            #         self.projection_per_seq[folder] = visible_cropped_downsampled_undistorted_projection_matrices
            #
            #         visible_cropped_downsampled_imgs = utils.get_color_imgs(folder, visible_view_indexes=
            #         self.visible_view_indexes_per_seq[folder],
            #                                                                 start_h=start_h, start_w=start_w,
            #                                                                 end_h=end_h, end_w=end_w,
            #                                                                 downsampling_factor=self.downsampling,
            #                                                                 is_hsv=self.is_hsv)
            #         # Calculate contaminated point list
            #         contaminated_point_list = utils.get_contaminated_point_list(imgs=visible_cropped_downsampled_imgs,
            #                                                                     point_cloud=self.point_cloud_per_seq[
            #                                                                         folder],
            #                                                                     mask_boundary=
            #                                                                     self.mask_boundary_per_seq[folder],
            #                                                                     inlier_percentage=self.inlier_percentage,
            #                                                                     projection_matrices=visible_cropped_downsampled_undistorted_projection_matrices,
            #                                                                     extrinsic_matrices=visible_extrinsic_matrices,
            #                                                                     is_hsv=self.is_hsv)
            #         self.contamination_point_list_per_seq[folder] = contaminated_point_list
            #         # Calculate appearing count of view per 3D point
            #         appearing_count_per_point = utils.get_visible_count_per_point(
            #             view_indexes_per_point=view_indexes_per_point)
            #         self.appearing_count_per_seq[folder] = appearing_count_per_point
            #
            with open(str(store_data_root / (phase + "_" +
                                             str(self.split_ratio[0]) + "_" + str(
                        self.split_ratio[1]) + "_" + str(
                        self.split_ratio[2]) + "_" + str(
                        self.downsampling) + "_" + str(
                        self.net_depth) + "_" + str(self.inlier_percentage) + ".pkl")), "wb") as f:
                pickle.dump(
                    [self.crop_positions_per_seq, self.selected_indexes_per_seq,
                     self.visible_view_indexes_per_seq,
                     self.point_cloud_per_seq, self.intrinsic_matrix_per_seq,
                     self.mask_boundary_per_seq, self.view_indexes_per_point_per_seq, self.extrinsics_per_seq,
                     self.projection_per_seq, self.contamination_point_list_per_seq,
                     self.appearing_count_per_seq, self.downsampling, self.net_depth, self.inlier_percentage],
                    f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(str(store_data_root / (phase + "_" +
                                             str(self.split_ratio[0]) + "_" + str(
                        self.split_ratio[1]) + "_" + str(
                        self.split_ratio[2]) + "_" + str(
                        self.downsampling) + "_" + str(
                        self.net_depth) + "_" + str(self.inlier_percentage) + ".pkl")), "rb") as f:

                [self.crop_positions_per_seq, self.selected_indexes_per_seq,
                 self.visible_view_indexes_per_seq,
                 self.point_cloud_per_seq, self.intrinsic_matrix_per_seq,
                 self.mask_boundary_per_seq, self.view_indexes_per_point_per_seq, self.extrinsics_per_seq,
                 self.projection_per_seq, self.contamination_point_list_per_seq,
                 self.appearing_count_per_seq, self.downsampling, self.net_depth,
                 self.inlier_percentage] = pickle.load(f)

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img_file_name = str(self.image_file_names[idx])
        # Retrieve the folder path
        folder = img_file_name[:-12]
        # Randomly pick one adjacent frame
        # We assume the filename has 8 logits followed by ".jpg"
        start_h, end_h, start_w, end_w = self.crop_positions_per_seq[folder]
        pos, increment = utils.generating_pos_and_increment(idx=idx,
                                                            visible_view_indexes=self.visible_view_indexes_per_seq[
                                                                folder],
                                                            adjacent_range=self.adjacent_range)
        # Get pair visible view indexes and pair extrinsic and projection matrices
        pair_indexes = [self.visible_view_indexes_per_seq[folder][pos],
                        self.visible_view_indexes_per_seq[folder][pos + increment]]
        pair_extrinsic_matrices = [self.extrinsics_per_seq[folder][pos],
                                   self.extrinsics_per_seq[folder][pos + increment]]
        pair_projection_matrices = [self.projection_per_seq[folder][pos],
                                    self.projection_per_seq[folder][pos + increment]]
        # Read pair images with downsampling and cropping
        pair_imgs = utils.get_pair_color_imgs(prefix_seq=folder, pair_indexes=pair_indexes, start_h=start_h,
                                              start_w=start_w,
                                              end_h=end_h, end_w=end_w, downsampling_factor=self.downsampling,
                                              is_hsv=self.is_hsv)
        pair_mask_imgs, pair_sparse_depth_imgs, pair_opt_flow_mask_imgs, pair_opt_flow_imgs = \
            utils.get_torch_training_data(pair_images=pair_imgs, pair_extrinsics=pair_extrinsic_matrices,
                                          pair_projections=
                                          pair_projection_matrices, pair_indexes=pair_indexes,
                                          point_cloud=self.point_cloud_per_seq[folder],
                                          mask_boundary=self.mask_boundary_per_seq[folder],
                                          view_indexes_per_point=self.view_indexes_per_point_per_seq[folder],
                                          visible_view_indexes=self.visible_view_indexes_per_seq[folder],
                                          contamination_point_list=self.contamination_point_list_per_seq[folder],
                                          appearing_count_per_point=self.appearing_count_per_seq[folder],
                                          use_view_indexes_per_point=self.use_view_indexes_per_point,
                                          visualize=self.visualize)
        # Calculate relative motion between two frames
        relative_motion = np.matmul(pair_extrinsic_matrices[0], np.linalg.inv(pair_extrinsic_matrices[1]))
        rotation = np.reshape(relative_motion[:3, :3], (3, 3))
        translation = np.reshape(relative_motion[:3, 3], (3, 1))

        # Format training data
        training_color_img_1 = pair_imgs[0]
        training_color_img_2 = pair_imgs[1]

        training_rotation = rotation
        training_rotation_inverse = np.transpose(rotation)
        training_translation = translation
        training_translation_inverse = np.matmul(-np.transpose(rotation), translation)
        training_rotation = training_rotation.astype("float32")
        training_rotation_inverse = training_rotation_inverse.astype("float32")
        training_translation = training_translation.astype("float32")
        training_translation_inverse = training_translation_inverse.astype("float32")
        training_rotation = training_rotation.reshape((3, 3))
        training_rotation_inverse = training_rotation_inverse.reshape((3, 3))
        training_translation = training_translation.reshape((3, 1))
        training_translation_inverse = training_translation_inverse.reshape((3, 1))

        training_sparse_depth_img_1 = pair_sparse_depth_imgs[0].astype("float32")
        training_sparse_depth_img_2 = pair_sparse_depth_imgs[1].astype("float32")
        training_mask_img_1 = pair_mask_imgs[0].astype("float32")
        training_mask_img_2 = pair_mask_imgs[1].astype("float32")
        training_sparse_depth_img_1 = training_sparse_depth_img_1.reshape((training_sparse_depth_img_1.shape[0],
                                                                           training_sparse_depth_img_1.shape[1], 1))
        training_sparse_depth_img_2 = training_sparse_depth_img_2.reshape((training_sparse_depth_img_2.shape[0],
                                                                           training_sparse_depth_img_2.shape[1], 1))
        training_mask_img_1 = training_mask_img_1.reshape(
            (training_mask_img_1.shape[0], training_mask_img_1.shape[1], 1))

        training_mask_img_2 = training_mask_img_2.reshape(
            (training_mask_img_2.shape[0], training_mask_img_2.shape[1], 1))

        training_opt_flow_mask_img_1 = pair_opt_flow_mask_imgs[0].astype("float32")
        training_opt_flow_mask_img_2 = pair_opt_flow_mask_imgs[1].astype("float32")
        training_opt_flow_img_1 = pair_opt_flow_imgs[0].astype("float32")
        training_opt_flow_img_2 = pair_opt_flow_imgs[1].astype("float32")

        training_intrinsic_matrix = self.intrinsic_matrix_per_seq[folder][:3, :3]
        training_intrinsic_matrix = training_intrinsic_matrix.astype("float32")
        training_intrinsic_matrix = training_intrinsic_matrix.reshape((3, 3))

        training_mask_boundary = self.mask_boundary_per_seq[folder].astype("float32") / 255.0
        training_mask_boundary[training_mask_boundary > 0.9] = 1.0
        training_mask_boundary[training_mask_boundary <= 0.9] = 0.0
        training_mask_boundary = training_mask_boundary.reshape(
            (training_mask_boundary.shape[0], training_mask_boundary.shape[1], 1))

        if self.to_augment:
            training_color_img_1, _ = self.transform(training_color_img_1)
            training_color_img_2, _ = self.transform(training_color_img_2)

        return [img_to_tensor(training_color_img_1), img_to_tensor(training_color_img_2),
                img_to_tensor(training_sparse_depth_img_1), img_to_tensor(training_sparse_depth_img_2),
                img_to_tensor(training_mask_img_1), img_to_tensor(training_mask_img_2),
                img_to_tensor(training_opt_flow_img_1), img_to_tensor(training_opt_flow_img_2),
                img_to_tensor(training_opt_flow_mask_img_1), img_to_tensor(training_opt_flow_mask_img_2),
                img_to_tensor(training_mask_boundary),
                torch.from_numpy(training_rotation),
                torch.from_numpy(training_rotation_inverse), torch.from_numpy(training_translation),
                torch.from_numpy(training_translation_inverse), torch.from_numpy(training_intrinsic_matrix)]
