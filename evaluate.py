'''
Author: Xingtong Liu, Ayushi Sinha, Masaru Ishii, Gregory D. Hager, Austin Reiter, Russell H. Taylor, and Mathias Unberath

Copyright (C) 2019 Johns Hopkins University - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU GENERAL PUBLIC LICENSE Version 3 license for non-commercial usage.

You should have received a copy of the GNU GENERAL PUBLIC LICENSE Version 3 license with
this file. If not, please write to: xliu89@jh.edu or rht@jhu.edu or unberath@jhu.edu
'''

import tqdm
import cv2
import numpy as np
from pathlib import Path
import torchsummary
import torch
import random
from tensorboardX import SummaryWriter
import albumentations as albu
import argparse
import datetime
# Local
import models
import utils
import dataset

if __name__ == '__main__':
    cv2.destroyAllWindows()
    parser = argparse.ArgumentParser(
        description='Self-supervised Depth Estimation on Monocular Endoscopy Dataset--Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_downsampling', type=float, default=4.0,
                        help='image downsampling rate to speed up training and reduce overfitting')
    parser.add_argument('--torchsummary_input_size', nargs='+', type=int, required=True,
                        help='input size for torchsummary (analysis purpose only)')
    parser.add_argument('--selected_frame_index_list', nargs='+', type=int, required=False, default=None,
                        help='selected frame index list)')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for testing')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for input data loader')
    parser.add_argument('--adjacent_range', nargs='+', type=int, required=True,
                        help='interval range for a pair of video frames')
    parser.add_argument('--id_range', nargs='+', type=int, required=True,
                        help='id range for the training and testing dataset')
    parser.add_argument('--network_downsampling', type=int, default=64, help='downsampling of network')
    parser.add_argument('--inlier_percentage', type=float, default=0.995,
                        help='percentage of inliers of SfM point clouds (for pruning some outliers)')
    parser.add_argument('--testing_patient_id', nargs='+', type=int, help='id of the testing patient')
    parser.add_argument('--load_intermediate_data', action='store_true', help='whether to load intermediate data')
    parser.add_argument('--use_hsv_colorspace', action='store_true',
                        help='convert RGB to hsv colorspace')
    parser.add_argument('--architecture_summary', action='store_true', help='display the network architecture')
    parser.add_argument('--load_all_frames', action='store_true',
                        help='whether or not to load all frames in sequence root')
    parser.add_argument('--trained_model_path', type=str, required=True, help='path to the trained student model')
    parser.add_argument('--sequence_root', type=str, required=True, help='path to the testing sequence')
    parser.add_argument('--evaluation_result_root', type=str, required=True,
                        help='root of the training input and ouput')
    parser.add_argument('--evaluation_data_root', type=str, required=True, help='path to the training data')
    parser.add_argument('--phase', type=str, required=True, help='evaluation phase')
    args = parser.parse_args()

    # Fix randomness for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(10085)
    np.random.seed(10085)
    random.seed(10085)

    # Hyper-parameters
    if args.torchsummary_input_size is not None and len(args.torchsummary_input_size) == 2:
        height, width = args.torchsummary_input_size
    else:
        height = 256
        width = 320
    adjacent_range = args.adjacent_range
    id_range = args.id_range
    input_downsampling = args.input_downsampling
    batch_size = args.batch_size
    num_workers = args.num_workers
    network_downsampling = args.network_downsampling
    inlier_percentage = args.inlier_percentage
    testing_patient_id = args.testing_patient_id
    load_intermediate_data = args.load_intermediate_data
    is_hsv = args.use_hsv_colorspace
    display_architecture = args.architecture_summary
    selected_frame_index_list = args.selected_frame_index_list
    load_all_frames = args.load_all_frames
    phase = args.phase
    evaluation_result_root = Path(args.evaluation_result_root)
    evaluation_data_root = Path(args.evaluation_data_root)
    trained_model_path = Path(args.trained_model_path)
    sequence_root = Path(args.sequence_root)
    currentDT = datetime.datetime.now()

    depth_estimation_model_teacher = []
    failure_sequences = []

    test_transforms = albu.Compose([
        albu.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, p=1.)], p=1.)

    log_root = Path(evaluation_result_root) / "depth_estimation_evaluation_run_{}_{}_{}_{}_test_id_{}".format(
        currentDT.month,
        currentDT.day,
        currentDT.hour,
        currentDT.minute,
        testing_patient_id)
    if not log_root.exists():
        log_root.mkdir(parents=True)
    writer = SummaryWriter(logdir=str(log_root))
    print("Tensorboard visualization at {}".format(str(log_root)))

    if selected_frame_index_list is None and not load_all_frames:
        raise IOError

    # Read all frame indexes
    if load_all_frames:
        selected_frame_index_list = utils.read_visible_view_indexes(sequence_root)

    # Get color image filenames
    test_filenames = utils.get_filenames_from_frame_indexes(sequence_root, selected_frame_index_list)
    folder_list = utils.get_parent_folder_names(evaluation_data_root, id_range=id_range)

    if phase == "validation":
        test_dataset = dataset.SfMDataset(image_file_names=test_filenames,
                                          folder_list=folder_list,
                                          adjacent_range=adjacent_range, transform=None,
                                          downsampling=input_downsampling,
                                          network_downsampling=network_downsampling,
                                          inlier_percentage=inlier_percentage,
                                          use_store_data=load_intermediate_data,
                                          store_data_root=evaluation_data_root,
                                          phase="validation", is_hsv=is_hsv,
                                          num_pre_workers=num_workers, visible_interval=30, rgb_mode="rgb")

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=0)
        depth_estimation_model = models.FCDenseNet57(n_classes=1)
        # Initialize the depth estimation network with Kaiming He initialization
        utils.init_net(depth_estimation_model, type="kaiming", mode="fan_in", activation_mode="relu",
                       distribution="normal")
        # Multi-GPU running
        depth_estimation_model = torch.nn.DataParallel(depth_estimation_model)
        # Summary network architecture
        if display_architecture:
            torchsummary.summary(depth_estimation_model, input_size=(3, height, width))

        # Load previous student model
        state = {}
        if trained_model_path.exists():
            print("Loading {:s} ...".format(str(trained_model_path)))
            state = torch.load(str(trained_model_path))
            step = state['step']
            epoch = state['epoch']
            depth_estimation_model.load_state_dict(state['model'])
            print('Restored model, epoch {}, step {}'.format(epoch, step))
        else:
            print("Trained model could not be found")
            raise OSError
        depth_estimation_model = depth_estimation_model.module

        # Custom layers
        depth_scaling_layer = models.DepthScalingLayer()
        depth_warping_layer = models.DepthWarpingLayer()
        flow_from_depth_layer = models.FlowfromDepthLayer()

        with torch.no_grad():
            # Set model to evaluation mode
            depth_estimation_model.eval()
            # Update progress bar
            tq = tqdm.tqdm(total=len(test_loader) * batch_size)
            for batch, (
                    colors_1, colors_2, sparse_depths_1, sparse_depths_2, sparse_depth_masks_1, sparse_depth_masks_2,
                    sparse_flows_1, sparse_flows_2, sparse_flow_masks_1, sparse_flow_masks_2, boundaries,
                    rotations_1_wrt_2, rotations_2_wrt_1, translations_1_wrt_2, translations_2_wrt_1, intrinsics,
                    folders) in enumerate(test_loader):
                colors_1 = colors_1.cuda()
                colors_2 = colors_2.cuda()
                sparse_depths_1 = sparse_depths_1.cuda()
                sparse_depths_2 = sparse_depths_2.cuda()
                sparse_depth_masks_1 = sparse_depth_masks_1.cuda()
                sparse_depth_masks_2 = sparse_depth_masks_2.cuda()
                sparse_flows_1 = sparse_flows_1.cuda()
                sparse_flows_2 = sparse_flows_2.cuda()
                boundaries = boundaries.cuda()
                rotations_1_wrt_2 = rotations_1_wrt_2.cuda()
                rotations_2_wrt_1 = rotations_2_wrt_1.cuda()
                translations_1_wrt_2 = translations_1_wrt_2.cuda()
                translations_2_wrt_1 = translations_2_wrt_1.cuda()
                intrinsics = intrinsics.cuda()

                tq.update(batch_size)

                colors_1 = boundaries * colors_1
                colors_2 = boundaries * colors_2
                sparse_flows_1 = sparse_flows_1 * boundaries
                sparse_flows_2 = sparse_flows_2 * boundaries

                predicted_depth_maps_1 = depth_estimation_model(colors_1)
                predicted_depth_maps_2 = depth_estimation_model(colors_2)
                scaled_depth_maps_1, normalized_scale_std_1 = depth_scaling_layer(
                    [torch.abs(predicted_depth_maps_1), sparse_depths_1, sparse_depth_masks_1])
                scaled_depth_maps_2, normalized_scale_std_2 = depth_scaling_layer(
                    [torch.abs(predicted_depth_maps_2), sparse_depths_2, sparse_depth_masks_2])

                depth_array = scaled_depth_maps_1[0].squeeze(dim=0).data.cpu().numpy()
                color_array = np.uint8(255 * cv2.cvtColor(
                    (colors_1[0].permute(1, 2, 0).data.cpu().numpy() + 1.0) * 0.5, cv2.COLOR_HSV2BGR_FULL))
                boundary_array = boundaries[0].squeeze(dim=0).data.cpu().numpy()
                intrinsic_array = intrinsics[0].data.cpu().numpy()

                # Sparse flow loss
                flows_from_depth_1 = flow_from_depth_layer(
                    [scaled_depth_maps_1, boundaries, translations_1_wrt_2, rotations_1_wrt_2,
                     intrinsics])
                flows_from_depth_2 = flow_from_depth_layer(
                    [scaled_depth_maps_2, boundaries, translations_2_wrt_1, rotations_2_wrt_1,
                     intrinsics])

                flows_from_depth_1 = flows_from_depth_1 * boundaries
                flows_from_depth_2 = flows_from_depth_2 * boundaries

                warped_depth_maps_2_to_1, intersect_masks_1 = depth_warping_layer(
                    [scaled_depth_maps_1, scaled_depth_maps_2, boundaries, translations_1_wrt_2, rotations_1_wrt_2,
                     intrinsics])
                warped_depth_maps_1_to_2, intersect_masks_2 = depth_warping_layer(
                    [scaled_depth_maps_2, scaled_depth_maps_1, boundaries, translations_2_wrt_1, rotations_2_wrt_1,
                     intrinsics])

                colors_1_display, sparse_depths_1_display, pred_depths_1_display, warped_depths_1_display, sparse_flows_1_display, dense_flows_1_display = \
                    utils.display_color_sparse_depth_dense_depth_warped_depth_sparse_flow_dense_flow(idx=1, step=step,
                                                                                                     writer=writer,
                                                                                                     colors_1=colors_1,
                                                                                                     sparse_depths_1=sparse_depths_1,
                                                                                                     pred_depths_1=scaled_depth_maps_1 * boundaries,
                                                                                                     warped_depths_2_to_1=warped_depth_maps_2_to_1,
                                                                                                     sparse_flows_1=sparse_flows_1,
                                                                                                     flows_from_depth_1=flows_from_depth_1,
                                                                                                     phase="Evaluation",
                                                                                                     is_return_image=True,
                                                                                                     color_reverse=True,
                                                                                                     boundaries=boundaries
                                                                                                     )
                colors_2_display, sparse_depths_2_display, pred_depths_2_display, warped_depths_2_display, sparse_flows_2_display, dense_flows_2_display = \
                    utils.display_color_sparse_depth_dense_depth_warped_depth_sparse_flow_dense_flow(idx=2, step=step,
                                                                                                     writer=writer,
                                                                                                     colors_1=colors_2,
                                                                                                     sparse_depths_1=sparse_depths_2,
                                                                                                     pred_depths_1=scaled_depth_maps_2 * boundaries,
                                                                                                     warped_depths_2_to_1=warped_depth_maps_1_to_2,
                                                                                                     sparse_flows_1=sparse_flows_2,
                                                                                                     flows_from_depth_1=flows_from_depth_2,
                                                                                                     phase="Evaluation",
                                                                                                     is_return_image=True,
                                                                                                     color_reverse=True,
                                                                                                     boundaries=boundaries
                                                                                                     )
                image_display = utils.stack_and_display(phase="Evaluation",
                                                        title="Results (c1, sd1, d1, wd1, sf1, df1, c2, sd2, d2, wd2, sf2, df2)",
                                                        step=step, writer=writer,
                                                        image_list=[colors_1_display, sparse_depths_1_display,
                                                                    pred_depths_1_display,
                                                                    warped_depths_1_display, sparse_flows_1_display,
                                                                    dense_flows_1_display,
                                                                    colors_2_display, sparse_depths_2_display,
                                                                    pred_depths_2_display,
                                                                    warped_depths_2_display, sparse_flows_2_display,
                                                                    dense_flows_2_display],
                                                        return_image=True)
                cv2.imwrite(str(log_root / "{}.png".format(batch)),
                            cv2.cvtColor(np.uint8(image_display * 255), cv2.COLOR_RGB2BGR))

                point_cloud = utils.point_cloud_from_depth(depth_array, color_array, boundary_array,
                                                           intrinsic_array, point_cloud_downsampling=1)
                utils.write_point_cloud(str(log_root / "{}.ply".format(batch)), point_cloud)

        tq.close()
        writer.close()

    elif phase == "test":
        test_dataset = dataset.SfMDataset(image_file_names=test_filenames,
                                          folder_list=folder_list,
                                          adjacent_range=adjacent_range, transform=None,
                                          downsampling=input_downsampling,
                                          network_downsampling=network_downsampling,
                                          inlier_percentage=inlier_percentage,
                                          use_store_data=load_intermediate_data,
                                          store_data_root=evaluation_data_root,
                                          phase="test", is_hsv=is_hsv,
                                          num_pre_workers=num_workers, visible_interval=20, rgb_mode="bgr")

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                                                  num_workers=0)
        depth_estimation_model = models.FCDenseNet57(n_classes=1)
        # Initialize the depth estimation network with Kaiming He initialization
        utils.init_net(depth_estimation_model, type="kaiming", mode="fan_in", activation_mode="relu",
                       distribution="normal")
        # Multi-GPU running
        depth_estimation_model = torch.nn.DataParallel(depth_estimation_model)
        # Summary network architecture
        if display_architecture:
            torchsummary.summary(depth_estimation_model, input_size=(3, height, width))

        # Load previous student model
        state = {}
        if trained_model_path.exists():
            print("Loading {:s} ...".format(str(trained_model_path)))
            state = torch.load(str(trained_model_path))
            step = state['step']
            epoch = state['epoch']
            depth_estimation_model.load_state_dict(state['model'])
            print('Restored model, epoch {}, step {}'.format(epoch, step))
        else:
            print("Trained model could not be found")
            raise OSError
        depth_estimation_model = depth_estimation_model.module

        with torch.no_grad():
            # Set model to evaluation mode
            depth_estimation_model.eval()
            # Update progress bar
            tq = tqdm.tqdm(total=len(test_loader) * batch_size)
            for batch, (colors_1, boundaries, intrinsics, names) in enumerate(test_loader):
                colors_1 = colors_1.cuda()
                boundaries = boundaries.cuda()

                colors_1 = boundaries * colors_1
                predicted_depth_maps_1 = depth_estimation_model(colors_1)

                color_display = cv2.cvtColor(np.uint8(
                    255 * (0.5 * colors_1[0].permute(1, 2, 0).data.cpu().numpy() + 0.5).reshape((height, width, 3))),
                    cv2.COLOR_HSV2BGR_FULL)
                boundary = boundaries[0].data.cpu().numpy().reshape((height, width))
                color_display = np.uint8(boundary.reshape((height, width, 1)) * color_display)
                depth_map = (boundaries * predicted_depth_maps_1)[0].data.cpu().numpy().reshape((height, width))
                depth_display = cv2.applyColorMap(np.uint8(255 * depth_map / np.max(depth_map)), cv2.COLORMAP_JET)
                point_cloud = utils.point_cloud_from_depth(depth_map=depth_map, color_img=color_display,
                                                           mask_img=boundary,
                                                           intrinsic_matrix=intrinsics[0].data.numpy(),
                                                           point_cloud_downsampling=1)
                utils.write_point_cloud(path=log_root / "{}.ply".format(names[0]), point_cloud=point_cloud)
                cv2.imwrite(str(log_root / "{}.png".format(names[0])), cv2.hconcat([color_display, depth_display]))
                tq.update(batch_size)
